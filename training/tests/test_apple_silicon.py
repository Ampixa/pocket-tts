"""Apple Silicon training path: device selection, Mimi placement, latent parity.

These run anywhere (the MPS-only cases skip without the backend) so a Linux CI
still exercises the CPU fallbacks.
"""

import os

import pytest
import torch

from training.distributed import mimi_device_for, mps_conv1d_length_limited, place_mimi


def _mps() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def test_mimi_device_for_passes_through_non_mps():
    assert mimi_device_for(torch.device("cpu")) == torch.device("cpu")
    assert mimi_device_for(torch.device("cuda", 0)) == torch.device("cuda", 0)


@pytest.mark.skipif(not _mps(), reason="needs the MPS backend")
def test_conv1d_limit_probe_matches_the_os():
    """The probe must agree with what MPS actually does on this OS."""
    try:
        torch.nn.functional.conv1d(
            torch.zeros(1, 1, 65537, device="mps"), torch.zeros(1, 1, 1, device="mps")
        )
        limited = False
    except NotImplementedError:
        limited = True
    assert mps_conv1d_length_limited(torch.device("mps")) == limited


@pytest.mark.skipif(not _mps(), reason="needs the MPS backend")
def test_chunked_streaming_encode_matches_whole_clip_encode():
    """Chunked-with-state must be the same computation as the whole clip, on
    CPU (exact up to float order) and on MPS (up to device numerics). This is
    what lets Mimi stay on MPS under the macOS 14 conv1d limit."""
    from training.modules.builders import load_model_config
    from training.modules.mimi_chunked import chunk_frames_for, encode_to_latent_chunked
    from training.scripts.precompute_latents import load_frozen_mimi

    config_path = os.environ.get("POCKET_TTS_TEST_MODEL_CONFIG")
    if not config_path:
        pytest.skip("set POCKET_TTS_TEST_MODEL_CONFIG to a model config with cached weights")
    mimi = load_frozen_mimi(load_model_config(config_path, {}))
    torch.manual_seed(0)
    audio = torch.randn(2, 1, 24000 * 9) * 0.1  # 9 s: several chunks
    frames = chunk_frames_for(mimi)
    assert frames * mimi.frame_size <= 65_536
    with torch.no_grad():
        cpu_full = mimi.to("cpu").encode_to_latent(audio)
        cpu_chunk = encode_to_latent_chunked(mimi, audio, frames)
        mps_chunk = encode_to_latent_chunked(mimi.to("mps"), audio.to("mps"), frames).cpu()
    rel = lambda a, b: float((a - b).norm() / (a.norm() + 1e-9))  # noqa: E731
    assert cpu_full.shape == cpu_chunk.shape == mps_chunk.shape
    assert rel(cpu_full, cpu_chunk) < 1e-5, f"cpu chunked differs by {rel(cpu_full, cpu_chunk):.2e}"
    assert rel(cpu_full, mps_chunk) < 1e-3, f"mps chunked differs by {rel(cpu_full, mps_chunk):.2e}"


@pytest.mark.skipif(not _mps(), reason="needs the MPS backend")
def test_place_mimi_keeps_mimi_on_mps():
    from training.modules.builders import load_model_config
    from training.scripts.precompute_latents import load_frozen_mimi

    config_path = os.environ.get("POCKET_TTS_TEST_MODEL_CONFIG")
    if not config_path:
        pytest.skip("set POCKET_TTS_TEST_MODEL_CONFIG to a model config with cached weights")
    mimi = load_frozen_mimi(load_model_config(config_path, {}))
    place_mimi(mimi, torch.device("mps"))
    assert next(mimi.parameters()).device.type == "mps"
    if mps_conv1d_length_limited(torch.device("mps")):
        assert getattr(mimi, "chunked_encode_frames", 0) > 0
    with torch.no_grad():
        out = mimi.encode_to_latent(torch.randn(1, 1, 24000 * 5, device="mps") * 0.1)
    assert out.device.type == "mps" and torch.isfinite(out).all()


@pytest.mark.skipif(not _mps(), reason="needs the MPS backend")
def test_encode_batch_lands_latents_on_the_model_device():
    """Mimi on CPU, FlowLM on MPS: encode_batch must hand MPS tensors back."""
    from training.dataloader.encode import encode_batch
    from training.dataloader.types import Batch
    from training.modules.builders import load_model_config
    from training.scripts.precompute_latents import load_frozen_mimi

    config_path = os.environ.get("POCKET_TTS_TEST_MODEL_CONFIG")
    if not config_path:
        pytest.skip("set POCKET_TTS_TEST_MODEL_CONFIG to a model config with cached weights")
    mimi = load_frozen_mimi(load_model_config(config_path, {})).to("cpu")  # Mimi on CPU, FlowLM on MPS
    sr = mimi.sample_rate
    audio = torch.randn(2, 1, sr * 2) * 0.1
    batch = Batch(
        audio=audio,
        num_audio_frames=torch.tensor([2 * mimi.frame_rate, mimi.frame_rate], dtype=torch.long),
        voice_audio=audio[:, :, : sr],
        num_voice_prompt_frames=torch.tensor([mimi.frame_rate, mimi.frame_rate], dtype=torch.long),
        text_tokens=[torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
        tail_latents=None,
        prompt_latents=None,
    )
    latents, mask, voice, n_voice = encode_batch(mimi, batch, torch.device("mps"))
    for t in (latents, mask, voice, n_voice):
        assert t.device.type == "mps"
    assert torch.isfinite(latents).all()


@pytest.mark.skipif(not _mps(), reason="needs the MPS backend")
def test_mimi_latents_agree_between_cpu_and_mps_on_a_short_clip():
    """Where MPS can run the encoder at all (< 65,536 samples), CPU and MPS
    latents must agree; otherwise the CPU fallback would be training on a
    different target than the model was built for."""
    from training.modules.builders import load_model_config
    from training.scripts.precompute_latents import load_frozen_mimi

    config_path = os.environ.get("POCKET_TTS_TEST_MODEL_CONFIG")
    if not config_path:
        pytest.skip("set POCKET_TTS_TEST_MODEL_CONFIG to a model config with cached weights")
    mimi = load_frozen_mimi(load_model_config(config_path, {}))
    torch.manual_seed(0)
    audio = torch.randn(1, 1, 60_000) * 0.1
    with torch.no_grad():
        cpu = mimi.to("cpu").encode_to_latent(audio)
        mps = mimi.to("mps").encode_to_latent(audio.to("mps")).cpu()
    assert cpu.shape == mps.shape
    rel = (cpu - mps).norm() / (cpu.norm() + 1e-9)
    assert rel < 1e-3, f"CPU vs MPS Mimi latents differ by {rel:.2e}"

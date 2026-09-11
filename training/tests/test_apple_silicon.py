"""Apple Silicon training path: device selection, Mimi placement, latent parity.

These run anywhere (the MPS-only cases skip without the backend) so a Linux CI
still exercises the CPU fallbacks.
"""

import os

import pytest
import torch

from training.distributed import mimi_device_for


def _mps() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def test_mimi_device_for_passes_through_non_mps():
    assert mimi_device_for(torch.device("cpu")) == torch.device("cpu")
    assert mimi_device_for(torch.device("cuda", 0)) == torch.device("cuda", 0)


@pytest.mark.skipif(not _mps(), reason="needs the MPS backend")
def test_mimi_device_for_matches_the_conv1d_probe():
    """The decision must agree with what MPS actually does on this OS."""
    device = mimi_device_for(torch.device("mps"))
    try:
        torch.nn.functional.conv1d(
            torch.zeros(1, 1, 65537, device="mps"), torch.zeros(1, 1, 1, device="mps")
        )
        supported = True
    except NotImplementedError:
        supported = False
    assert device == (torch.device("mps") if supported else torch.device("cpu"))


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
    mimi = load_frozen_mimi(load_model_config(config_path, {})).to("cpu")
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

"""Chunked Mimi encoding for MPS on macOS 14, which rejects conv1d inputs longer
than 65,536 samples.

Mimi's encoder is a streaming model: every conv keeps `kernel - stride` samples
of history in its state and the transformer keeps a KV cache with the same
context window the non-streaming mask uses, so encoding a waveform in chunks
with carried state is the same computation as encoding it whole. This module
does that with chunks that stay under the MPS limit. `test_apple_silicon.py`
checks the chunked result against the whole-clip CPU encode.
"""

from __future__ import annotations

import types

import torch

from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules.conv import pad_for_conv1d
from pocket_tts.modules.stateful_module import StatefulModule, increment_steps, init_states

MPS_CONV1D_MAX_SAMPLES = 65_536


def chunk_frames_for(mimi: MimiModel, max_samples: int = MPS_CONV1D_MAX_SAMPLES) -> int:
    """Frames per chunk: as many whole codec frames as fit under the limit."""
    frames = max_samples // mimi.frame_size
    if frames < 1:
        raise ValueError(f"frame size {mimi.frame_size} exceeds the chunk limit {max_samples}")
    return frames


def _stamp(mimi: MimiModel) -> None:
    """Stateful modules look their state up by absolute name; init_states keys
    them by name relative to the module it was given, so stamp the same names.
    (TTSModel.stamp_state_names does this from the TTS root with a "mimi."
    prefix; a bare Mimi needs the relative form.) Idempotent."""
    for name, module in mimi.named_modules():
        if isinstance(module, StatefulModule):
            module._module_absolute_name = name


@torch.no_grad()
def encode_to_latent_chunked(mimi: MimiModel, x: torch.Tensor, chunk_frames: int) -> torch.Tensor:
    """Same contract as MimiModel.encode_to_latent: [B, C, T] audio -> [B, T', D] latents."""
    if x.dim() != 3:
        raise ValueError(f"expected audio of shape [B, C, T], got {tuple(x.shape)}")
    frame_size = mimi.frame_size
    x = pad_for_conv1d(x, frame_size, frame_size)
    total_frames = x.shape[-1] // frame_size
    # The KV cache and the step counter live at the encoder transformer's rate
    # (200 Hz), not the latent rate (12.5 Hz): size and advance them in those units,
    # exactly as the streaming decoder does (increment = ratio * latent frames).
    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    _stamp(mimi)
    state = init_states(mimi, x.shape[0], total_frames * ratio)
    outputs = []
    for start in range(0, total_frames, chunk_frames):
        piece = x[..., start * frame_size : (start + chunk_frames) * frame_size]
        emb = mimi.encoder(piece, state)
        (emb,) = mimi.encoder_transformer(emb, state)
        encoder_frames = emb.shape[-1]
        if mimi.encoder_frame_rate != mimi.frame_rate:
            emb = mimi.downsample(emb, state)
        increment_steps(mimi, state, encoder_frames)
        outputs.append(emb)
    return torch.cat(outputs, dim=-1).transpose(-1, -2)


def enable_chunked_encode(mimi: MimiModel, max_samples: int = MPS_CONV1D_MAX_SAMPLES) -> int:
    """Rebind mimi.encode_to_latent to the chunked path. Returns frames per chunk."""
    frames = chunk_frames_for(mimi, max_samples)

    def encode(self: MimiModel, x: torch.Tensor) -> torch.Tensor:
        return encode_to_latent_chunked(self, x, frames)

    mimi.encode_to_latent = types.MethodType(encode, mimi)  # type: ignore[method-assign]
    mimi.chunked_encode_frames = frames  # type: ignore[attr-defined]
    return frames

"""Stored cold stitches from different calibration lengths must not shift targets."""

from __future__ import annotations

import random

import torch

from training.dataloader.encode import encode_batch
from training.dataloader.loader import DataLoader
from training.dataloader.types import Entry


class ForbiddenMimi(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def encode_to_latent(self, _audio: torch.Tensor) -> torch.Tensor:
        raise AssertionError("stored latents must not invoke Mimi")


def test_mixed_stitch_lengths_preserve_each_target() -> None:
    loader = object.__new__(DataLoader)
    loader.stored_stitches = True
    short = torch.full((35, 2), 1.0)
    long = torch.full((36, 2), 3.0)
    short_tail = torch.full((5, 2), 2.0)
    long_tail = torch.full((4, 2), 4.0)
    samples = [
        (short, torch.tensor([1]), torch.full((5, 2), 10.0), short_tail, 40),
        (long, torch.tensor([2]), torch.full((4, 2), 11.0), long_tail, 40),
    ]

    batch = loader._collate_latent(samples)
    latents, mask, prompts, prompt_lengths = encode_batch(
        ForbiddenMimi(), batch, torch.device("cpu")
    )

    assert torch.equal(latents[0], torch.cat([short, short_tail]))
    assert torch.equal(latents[1], torch.cat([long, long_tail]))
    assert mask.shape == (2, 40) and mask.all()
    assert prompts.shape == (2, 5, 2)
    assert prompt_lengths.tolist() == [5, 4]


def test_stored_sample_uses_its_own_stitch_length() -> None:
    loader = object.__new__(DataLoader)
    loader.stitch_frames = 35  # package metadata from the other source
    loader.rng = random.Random(0)
    loader.tokenize = lambda text: [1] if text else []
    loader.frame_rate = 12.5
    loader.max_duration_sec = 20.0
    loader.max_voice_prompt_sec = 5.0
    loader._load_latents_with_stitches = lambda _path: (
        torch.ones(100, 2), torch.tensor([20]), torch.tensor([1]),
        torch.full((1, 36, 2), 3.0),
    )
    entry = Entry(
        path="/audio/is/not/needed.wav", duration=8.0, transcript="one two",
        words=[{"word": "one", "start": 0.0, "end": 1.0},
               {"word": "two", "start": 2.0, "end": 7.5}],
        latents_file="kid.safetensors",
    )

    stitch, tokens, prompt, tail, target_frames = loader._sample_latent_stored(entry)

    assert stitch.shape == (36, 2)
    assert len(tokens) == 1
    assert prompt.shape[1] == 2
    assert stitch.shape[0] + tail.shape[0] == target_frames

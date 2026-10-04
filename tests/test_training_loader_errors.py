"""Loader failures must reach the training loop instead of hanging or skipping."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from training.dataloader.loader import DataLoader, _prefetch


def test_prefetch_propagates_worker_exception() -> None:
    def broken():
        yield 1
        raise ValueError("bad manifest row")

    batches = _prefetch(broken())
    assert next(batches) == 1
    with pytest.raises(RuntimeError, match="data loader prefetch failed") as error:
        next(batches)
    assert isinstance(error.value.__cause__, ValueError)


def test_prefetch_close_releases_infinite_source() -> None:
    source_closed = threading.Event()

    def infinite():
        try:
            while True:
                yield 1
        finally:
            source_closed.set()

    batches = _prefetch(infinite(), depth=1)
    assert next(batches) == 1
    batches.close()
    assert source_closed.wait(timeout=2), "prefetch producer stayed blocked after close"
    assert not any(t.name == "tts-data-prefetch" and t.is_alive() for t in threading.enumerate())


def test_unreadable_sample_is_not_skipped() -> None:
    loader = object.__new__(DataLoader)
    loader.jsonl = "train.jsonl"

    def fail(_entry: object):
        raise FileNotFoundError("missing wav")

    loader._sample = fail
    entry = type("EntryFixture", (), {"path": "/missing/example.wav"})()

    with pytest.raises(RuntimeError, match="train.jsonl: failed to load /missing/example.wav") as error:
        loader._sample_checked(entry)
    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_missing_audio_reaches_batch_consumer(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "path": str(tmp_path / "missing.wav"),
                "duration": 2.0,
                "transcript": "कुकुरलाई पानी पिलायो",
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    loader = DataLoader(
        jsonl=str(manifest),
        tokenize=lambda text: [1] if text else [],
        batch_size=1,
        sample_rate=24000,
        frame_rate=12.5,
        max_duration_sec=20.0,
        max_voice_prompt_sec=5.0,
        rank=0,
        world_size=1,
        io_workers=1,
    )

    with pytest.raises(RuntimeError, match="data loader prefetch failed") as error:
        next(iter(loader))
    assert isinstance(error.value.__cause__, RuntimeError)
    assert "missing.wav" in str(error.value.__cause__)

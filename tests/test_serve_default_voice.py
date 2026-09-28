"""Tests for the default voice of the `serve` command and of the /tts endpoint."""

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from pocket_tts import main

runner = CliRunner()


class FakeTTSModel:
    """Stand-in for TTSModel, recording which voices and states it was asked for."""

    def __init__(self):
        self.config = SimpleNamespace(mimi=SimpleNamespace(sample_rate=24000))
        self.voices_requested: list[Path | str | torch.Tensor] = []
        self.states_used: list[dict[str, Any]] = []

    def get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
    ) -> dict[str, Any]:
        self.voices_requested.append(audio_conditioning)
        return {"voice": audio_conditioning}

    def _cached_get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
    ) -> dict[str, Any]:
        return self.get_state_for_audio_prompt(audio_conditioning, truncate)

    def generate_audio_stream(
        self, model_state: dict[str, Any], text_to_generate: str
    ) -> Iterator[torch.Tensor]:
        self.states_used.append(model_state)
        yield torch.zeros(2400)


def make_serve_runnable(monkeypatch: pytest.MonkeyPatch) -> FakeTTSModel:
    """Make `serve` return right before listening, with a fake model."""
    fake_model = FakeTTSModel()
    monkeypatch.setattr(main, "TTSModel", SimpleNamespace(load_model=lambda **kwargs: fake_model))
    monkeypatch.setattr(main.uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "tts_model", None)
    monkeypatch.setattr(main, "default_voice_state", None)
    return fake_model


def test_serve_loads_the_voice_given_by_the_default_voice_option(monkeypatch: pytest.MonkeyPatch):
    fake_model = make_serve_runnable(monkeypatch)

    result = runner.invoke(main.cli_app, ["serve", "--default-voice", "./my_voice.safetensors"])

    assert result.exit_code == 0, result.output
    assert fake_model.voices_requested == ["./my_voice.safetensors"]
    assert main.default_voice_state == {"voice": "./my_voice.safetensors"}


def test_serve_falls_back_to_the_voice_of_the_language(monkeypatch: pytest.MonkeyPatch):
    fake_model = make_serve_runnable(monkeypatch)

    result = runner.invoke(main.cli_app, ["serve", "--language", "french_24l"])

    assert result.exit_code == 0, result.output
    assert fake_model.voices_requested == ["estelle"]
    assert main.default_voice_state == {"voice": "estelle"}


def test_tts_endpoint_uses_the_default_voice_when_the_request_has_none(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_model = FakeTTSModel()
    monkeypatch.setattr(main, "tts_model", fake_model)
    monkeypatch.setattr(main, "default_voice_state", {"voice": "./my_voice.wav"})

    response = TestClient(main.web_app).post("/tts", data={"text": "Hello world."})

    assert response.status_code == 200
    # The default voice is served as-is, without being encoded again.
    assert fake_model.voices_requested == []
    assert fake_model.states_used == [{"voice": "./my_voice.wav"}]


def test_tts_endpoint_prefers_the_voice_of_the_request_over_the_default(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_model = FakeTTSModel()
    monkeypatch.setattr(main, "tts_model", fake_model)
    monkeypatch.setattr(main, "default_voice_state", {"voice": "./my_voice.wav"})

    response = TestClient(main.web_app).post(
        "/tts", data={"text": "Hello world.", "voice_url": "marius"}
    )

    assert response.status_code == 200
    assert fake_model.voices_requested == ["marius"]
    assert fake_model.states_used == [{"voice": "marius"}]


def test_tts_endpoint_still_rejects_a_voice_url_that_is_not_a_voice(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_model = FakeTTSModel()
    monkeypatch.setattr(main, "tts_model", fake_model)
    monkeypatch.setattr(main, "default_voice_state", {"voice": "./my_voice.wav"})

    response = TestClient(main.web_app).post(
        "/tts", data={"text": "Hello world.", "voice_url": "./my_voice.wav"}
    )

    assert response.status_code == 400
    assert fake_model.states_used == []


def test_serve_loads_local_voice_presets_and_serves_previews(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    fake_model = make_serve_runnable(monkeypatch)
    audio = tmp_path / "kore.wav"
    audio.write_bytes(b"RIFF-preset-test")
    manifest = tmp_path / "voices.json"
    manifest.write_text(json.dumps([{
        "id": "kore", "name": "Kore", "register": "conversation", "file": "kore.wav"
    }]), encoding="utf-8")

    result = runner.invoke(main.cli_app, [
        "serve", "--default-voice", str(audio), "--voice-presets-manifest", str(manifest)
    ])

    assert result.exit_code == 0, result.output
    assert fake_model.voices_requested == [str(audio), audio]
    client = TestClient(main.web_app)
    listed = client.get("/voice-presets")
    assert listed.status_code == 200
    assert listed.json() == [{
        "id": "kore", "name": "Kore", "register": "conversation",
        "preview_url": "voice-presets/kore/audio"
    }]
    preview = client.get("/voice-presets/kore/audio")
    assert preview.status_code == 200
    assert preview.content == b"RIFF-preset-test"
    generated = client.post("/tts", data={"text": "नमस्ते", "voice_preset": "kore"})
    assert generated.status_code == 200
    assert fake_model.states_used == [{"voice": audio}]


def test_preset_endpoint_rejects_unknown_and_ambiguous_voice(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_model = FakeTTSModel()
    monkeypatch.setattr(main, "tts_model", fake_model)
    monkeypatch.setattr(main, "preset_voice_states", {"kore": {"voice": "kore"}})
    client = TestClient(main.web_app)

    assert client.post("/tts", data={"text": "hello", "voice_preset": "missing"}).status_code == 400
    assert client.post("/tts", data={
        "text": "hello", "voice_preset": "kore", "voice_url": "marius"
    }).status_code == 400
    assert client.get("/voice-presets/missing/audio").status_code == 404
    assert fake_model.states_used == []


def test_preset_manifest_cannot_escape_its_directory(tmp_path: Path):
    bank = tmp_path / "bank"
    bank.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    manifest = bank / "voices.json"
    manifest.write_text(json.dumps([{
        "id": "outside", "name": "Outside", "register": "test", "file": "../outside.wav"
    }]), encoding="utf-8")

    with pytest.raises(ValueError, match="inside the bank"):
        main.load_voice_presets(str(manifest), FakeTTSModel())


def test_generation_error_reaches_the_response(monkeypatch: pytest.MonkeyPatch):
    class BrokenModel(FakeTTSModel):
        def generate_audio_stream(
            self, model_state: dict[str, Any], text_to_generate: str
        ) -> Iterator[torch.Tensor]:
            raise RuntimeError("codec failed")

    monkeypatch.setattr(main, "tts_model", BrokenModel())
    monkeypatch.setattr(main, "default_voice_state", {"voice": "test"})

    with pytest.raises(RuntimeError, match="codec failed"):
        TestClient(main.web_app).post("/tts", data={"text": "नमस्ते"})

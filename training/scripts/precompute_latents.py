import hashlib
import json
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import numpy.typing as npt
import safetensors.torch
import torch
import typer
from tqdm import tqdm

from pocket_tts.models.mimi import MimiModel
from pocket_tts.utils.config import Config
from pocket_tts.utils.utils import download_if_necessary
from training.args import load_args
from training.dataloader import Entry, _load_window
from training.dataloader.loader import cut_to_frames, eligible_cuts, latent_target_frames
from training.modules.builders import build_mimi, load_model_config

logger = logging.getLogger("precompute_latents")
app = typer.Typer(pretty_exceptions_show_locals=False)

CALIBRATION_POOL_LINES = 4096
CALIBRATION_MARGIN_FRAMES = 4


def default_decode_workers() -> int:
    """Size the decode pool from the cores this process may actually use."""
    return max(4, (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)) - 2)


# (path, start_sec, duration_sec, sample_rate): the arguments of _load_window.
DecodeJob = tuple[str, float, float, int]


def _decode_one(job: DecodeJob) -> npt.NDArray[np.float32]:
    path, start, duration, sample_rate = job
    return _load_window(path, start, duration, sample_rate)


def _decode_chunk(jobs: list[DecodeJob]) -> tuple[npt.NDArray[np.float32], list[int]]:
    wavs = [_load_window(p, s, d, sr) for (p, s, d, sr) in jobs]
    max_len = max(len(w) for w in wavs)
    batch = np.zeros((len(wavs), 1, max_len), dtype=np.float32)
    for b, w in enumerate(wavs):
        batch[b, 0, : len(w)] = w
    return batch, [len(w) for w in wavs]


def _parse_entry(line: str) -> Entry:
    d = json.loads(line)
    return Entry(
        d["path"], float(d["duration"]), d["transcript"], d.get("words"), float(d.get("start", 0.0))
    )


def _chunk_jobs(lines: list[str], idxs: list[int], sample_rate: int) -> list[DecodeJob]:
    chunk = [lines[i] for i in idxs]
    return [(e.path, e.start, e.duration, sample_rate) for e in map(_parse_entry, chunk)]


def mimi_encode_hash(mimi: MimiModel) -> str:
    """Hash of the weights on the encode path (encoder, encoder transformer,
    downsample): identifies which Mimi produced a latents store."""
    h = hashlib.sha256()
    for name in ("encoder", "encoder_transformer", "downsample"):
        module = getattr(mimi, name, None)
        if module is None:
            continue
        for k, v in sorted(module.state_dict().items()):
            h.update(f"{name}.{k}:{tuple(v.shape)}".encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_frozen_mimi(config: Config) -> MimiModel:
    mimi = build_mimi(config.mimi)
    weights_file = download_if_necessary(str(config.weights_path))
    state = safetensors.torch.load_file(weights_file)
    mimi_state = {k.removeprefix("mimi."): v for k, v in state.items() if k.startswith("mimi.")}
    mimi.load_state_dict(mimi_state, strict=True)
    encoder_max = max(
        v.abs().max().item() for k, v in mimi_state.items() if k.startswith("encoder.")
    )
    if encoder_max == 0:
        raise SystemExit(
            f"{config.weights_path} ships an all-zero Mimi encoder (a release without "
            "voice cloning). Point the config at weights with a real encoder."
        )
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad_(False)
    # Same per-module compile as training; dynamic shapes because chunks pad
    # to their own longest row.
    mimi.encoder.compile(dynamic=True)
    mimi.encoder_transformer.compile(dynamic=True)
    return mimi


@torch.no_grad()
def measure_stitch_frames(mimi: MimiModel, audio: torch.Tensor) -> tuple[int, float]:
    fs = mimi.frame_size
    full = mimi.encode_to_latent(audio)
    k = full.shape[1] // 2
    cold = mimi.encode_to_latent(audio[..., k * fs :])
    n = min(cold.shape[1], full.shape[1] - k) - 1
    rel = (cold[:, :n] - full[:, k : k + n]).norm(dim=-1) / (full[:, k : k + n].norm(dim=-1) + 1e-8)
    rel = rel.max(dim=0).values
    prompt_cold = mimi.encode_to_latent(audio[..., : k * fs])
    pn = min(prompt_cold.shape[1], k) - 1
    floor = (
        ((prompt_cold[:, :pn] - full[:, :pn]).norm(dim=-1) / (full[:, :pn].norm(dim=-1) + 1e-8))
        .max()
        .item()
    )
    above = (rel > max(3 * floor, 1e-3)).nonzero()
    frames = int(above.max().item()) + 1 if above.numel() else 0
    return frames + CALIBRATION_MARGIN_FRAMES, floor


def _calibrate(
    pool: ProcessPoolExecutor,
    lines: list[str],
    mimi: MimiModel,
    batch_size: int,
    device: torch.device,
) -> tuple[int, float]:
    longest = sorted(map(_parse_entry, lines[:CALIBRATION_POOL_LINES]), key=lambda e: -e.duration)
    jobs = [(e.path, e.start, e.duration, mimi.sample_rate) for e in longest[:batch_size]]
    calib = list(pool.map(_decode_one, jobs))
    max_len = max(len(w) for w in calib)
    max_len -= max_len % mimi.frame_size
    audio = torch.zeros(len(calib), 1, max_len)
    for b, w in enumerate(calib):
        audio[b, 0, : min(len(w), max_len)] = torch.from_numpy(w[:max_len])
    return measure_stitch_frames(mimi, audio.to(device))


def _entry_frames(n_samples: int, sample_rate: int, frame_rate: float) -> int:
    return max(1, int(n_samples * frame_rate / sample_rate))


def _atomic_write_text(path: Path, text: str):
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(text)
    tmp.rename(path)


def _latents_name(manifest: Path, idx: int, tag: str) -> str:
    return f"latents/{tag}/{manifest.stem}_{idx:08d}.safetensors"


def _annotated_lines(lines: list[str], manifest: Path, tag: str) -> list[str]:
    out = []
    for idx, line in enumerate(lines):
        d = json.loads(line)
        d["latents_file"] = _latents_name(manifest, idx, tag)
        out.append(json.dumps(d))
    return out


def _pending_chunks(
    lines: list[str],
    manifest: Path,
    batch_size: int,
    tag: str,
    worker: int = 0,
    num_workers: int = 1,
) -> list[list[int]]:
    # Chunk in duration order: a chunk pads to its longest row, so grouping
    # similar lengths avoids spending encode FLOPs on padding. Latents files
    # are named by manifest index, so encode order is free. Workers take
    # strided chunks, so several GPUs can encode one manifest concurrently.
    order = sorted(range(len(lines)), key=lambda i: json.loads(lines[i])["duration"])
    pending = []
    for n, chunk_start in enumerate(range(0, len(order), batch_size)):
        if n % num_workers != worker:
            continue
        idxs = order[chunk_start : chunk_start + batch_size]
        if any(not (manifest.parent / _latents_name(manifest, idx, tag)).exists() for idx in idxs):
            pending.append(idxs)
    return pending


def _stitch_windows(
    lens: list[int],
    idxs: list[int],
    lines: list[str],
    max_frames: int,
    stitch_frames: int,
    stitch_cuts: int,
    max_voice_prompt_sec: float,
    max_duration_sec: float,
    mimi: MimiModel,
) -> list[tuple[list[int], list[int], list]]:
    """Per row: (cut frames, first-word indices, audio windows) for K cuts.

    The window is read with the loader's own call -- `_load_window` at
    `start + cut/frame_rate` for `min(S, target)/frame_rate` seconds -- and
    zero-padded to S frames of samples the way the loader's collate does. Not
    sliced from the decoded chunk: that call has floating-point quirks (a start
    that can land one sample early, a length that truncates to 67,199 samples)
    which the teacher was trained through, so the stored stitch must reproduce
    them, not correct them. K cuts are spread evenly over the eligible
    boundaries so short and long prompts are both represented.
    """
    fs = mimi.frame_size
    out = []
    for b, n_samples in enumerate(lens):
        entry = _parse_entry(lines[idxs[b]])
        stored = min(_entry_frames(n_samples, mimi.sample_rate, mimi.frame_rate), max_frames)
        cuts = eligible_cuts(entry, max_voice_prompt_sec) if stored > 1 else []
        if len(cuts) > stitch_cuts:
            picks = sorted({round(j * (len(cuts) - 1) / (stitch_cuts - 1))
                            for j in range(stitch_cuts)}) if stitch_cuts > 1 else [0]
            cuts = [cuts[p] for p in picks]
        frames, words, windows = [], [], []
        for cut_sec, word_index in cuts:
            cut_frames = cut_to_frames(cut_sec, mimi.frame_rate, stored)
            target = latent_target_frames(
                entry, cut_frames, stored, mimi.frame_rate, max_duration_sec
            )
            row_stitch = min(stitch_frames, target)
            piece = _load_window(
                entry.path,
                entry.start + cut_frames / mimi.frame_rate,
                row_stitch / mimi.frame_rate,
                mimi.sample_rate,
            )
            window = np.zeros(stitch_frames * fs, dtype=np.float32)
            n = min(len(piece), stitch_frames * fs)
            window[:n] = piece[:n]
            frames.append(cut_frames)
            words.append(word_index)
            windows.append(window)
        out.append((frames, words, windows))
    return out


def _write_chunk(
    latents: torch.Tensor,
    lens: list[int],
    idxs: list[int],
    manifest: Path,
    mimi: MimiModel,
    tag: str,
    stitches: list[tuple[list[int], list[int], torch.Tensor]] | None = None,
):
    for b, n_samples in enumerate(lens):
        frames = min(_entry_frames(n_samples, mimi.sample_rate, mimi.frame_rate), latents.shape[1])
        path = manifest.parent / _latents_name(manifest, idxs[b], tag)
        payload = {"latents": latents[b, :frames].contiguous()}
        if stitches is not None:
            cut_frames, word_index, stitch_latents = stitches[b]
            if cut_frames:
                payload["stitch_cuts"] = torch.tensor(cut_frames, dtype=torch.long)
                payload["stitch_words"] = torch.tensor(word_index, dtype=torch.long)
                payload["stitch_latents"] = stitch_latents.contiguous()
        # Writer-unique tmp name: concurrent jobs racing on the same manifest
        # then only ever rename complete files (rename is atomic).
        tmp = path.with_suffix(f".tmp.{os.getpid()}")
        safetensors.torch.save_file(payload, str(tmp))
        tmp.rename(path)


def _encode_pending(
    pool: ProcessPoolExecutor,
    mimi: MimiModel,
    device: torch.device,
    lines: list[str],
    manifest: Path,
    batch_size: int,
    decode_workers: int,
    tag: str,
    worker: int = 0,
    num_workers: int = 1,
    stitch_frames: int = 0,
    stitch_cuts: int = 0,
    max_voice_prompt_sec: float = 5.0,
    max_duration_sec: float = 20.0,
):
    pending = _pending_chunks(lines, manifest, batch_size, tag, worker, num_workers)
    lookahead = decode_workers + 2  # keep every decode worker busy
    futures = {
        i: pool.submit(_decode_chunk, _chunk_jobs(lines, idxs, mimi.sample_rate))
        for i, idxs in enumerate(pending[:lookahead])
    }
    submitted = len(futures)
    for i, idxs in enumerate(tqdm(pending, desc=f"encode {manifest.name}")):
        arr, lens = futures.pop(i).result()
        if submitted < len(pending):
            jobs = _chunk_jobs(lines, pending[submitted], mimi.sample_rate)
            futures[submitted] = pool.submit(_decode_chunk, jobs)
            submitted += 1
        with torch.no_grad():
            latents = mimi.encode_to_latent(torch.from_numpy(arr).to(device)).cpu()
            stitches = None
            if stitch_cuts > 0:
                rows = _stitch_windows(
                    lens, idxs, lines, latents.shape[1], stitch_frames, stitch_cuts,
                    max_voice_prompt_sec, max_duration_sec, mimi,
                )
                flat = [w for _, _, windows in rows for w in windows]
                encoded = None
                if flat:
                    # One batched call: every window starts from a fresh state,
                    # which is exactly what makes these cold stitches.
                    batch = torch.from_numpy(np.stack(flat))[:, None, :].to(device)
                    encoded = mimi.encode_to_latent(batch).cpu()[:, :stitch_frames]
                stitches, offset = [], 0
                for frames, words, windows in rows:
                    n = len(windows)
                    lat = encoded[offset : offset + n] if n else torch.zeros(0, stitch_frames, latents.shape[-1])
                    stitches.append((frames, words, lat))
                    offset += n
        _write_chunk(latents, lens, idxs, manifest, mimi, tag, stitches)


def _write_manifest_and_meta(
    manifest: Path,
    new_lines: list[str],
    stitch_frames: int,
    floor: float,
    mimi: MimiModel,
    weights_path: str,
    mimi_hash: str,
    stitch_cuts: int = 0,
    max_voice_prompt_sec: float = 5.0,
    max_duration_sec: float = 20.0,
):
    out_manifest = manifest.with_name(manifest.stem + "_latents.jsonl")
    _atomic_write_text(out_manifest, "\n".join(new_lines) + "\n")
    meta = {
        "stitch_frames": stitch_frames,
        "noise_floor": floor,
        "frame_rate": mimi.frame_rate,
        "weights_path": weights_path,
        "mimi_hash": mimi_hash,
        # > 0 means every row carries cold stitches at this many candidate cuts,
        # chosen with this prompt window, and the loader needs no audio.
        "stitch_cuts": stitch_cuts,
        "max_voice_prompt_sec": max_voice_prompt_sec,
        "max_duration_sec": max_duration_sec,
    }
    meta_path = manifest.with_name(manifest.stem + "_latents.meta.json")
    _atomic_write_text(meta_path, json.dumps(meta, indent=2) + "\n")
    logger.info(f"wrote {out_manifest} and {meta_path}")


def precompute_manifest(
    manifest: Path,
    mimi: MimiModel,
    device: torch.device,
    batch_size: int,
    decode_workers: int,
    weights_path: str,
    worker: int = 0,
    num_workers: int = 1,
    stitch_cuts: int = 0,
    max_voice_prompt_sec: float = 5.0,
    max_duration_sec: float = 20.0,
):
    """Encode a manifest's utterances to per-utterance latents files.

    With num_workers > 1 each worker encodes a strided subset of chunks;
    worker 0 waits for the others' files and writes the manifest and meta.

    stitch_cuts > 0 also stores cold stitch latents at that many candidate
    cuts per row, which lets the loader train with no audio present. Such a
    store gets its own tag so it cannot be confused with one that lacks them.
    """
    lines = manifest.read_text().splitlines()
    mimi_hash = mimi_encode_hash(mimi)
    tag = mimi_hash[:8] + (f"s{stitch_cuts}" if stitch_cuts > 0 else "")
    (manifest.parent / "latents" / tag).mkdir(parents=True, exist_ok=True)
    decode_workers = decode_workers or default_decode_workers()
    pool = ProcessPoolExecutor(
        max_workers=decode_workers, mp_context=multiprocessing.get_context("spawn")
    )
    stitch_frames, floor = 0, 0.0
    if worker == 0 or stitch_cuts > 0:
        # Every worker needs stitch_frames when it is storing stitches. The
        # calibration pool is deterministic, so they all measure the same value.
        stitch_frames, floor = _calibrate(pool, lines, mimi, batch_size, device)
        logger.info(f"{manifest.name}: stitch_frames={stitch_frames} (noise floor {floor:.1e})")
    _encode_pending(
        pool, mimi, device, lines, manifest, batch_size, decode_workers, tag, worker, num_workers,
        stitch_frames=stitch_frames, stitch_cuts=stitch_cuts,
        max_voice_prompt_sec=max_voice_prompt_sec, max_duration_sec=max_duration_sec,
    )
    if worker != 0:
        return
    while _pending_chunks(lines, manifest, batch_size, tag):
        time.sleep(5)
    new_lines = _annotated_lines(lines, manifest, tag)
    _write_manifest_and_meta(
        manifest, new_lines, stitch_frames, floor, mimi, weights_path, mimi_hash,
        stitch_cuts=stitch_cuts, max_voice_prompt_sec=max_voice_prompt_sec,
        max_duration_sec=max_duration_sec,
    )


@app.command()
def main(
    config: str,
    batch_size: int = 16,
    decode_workers: int = 0,
    stitch_cuts: int = 0,
):
    logging.basicConfig(level=logging.INFO)
    args = load_args(config)
    model_config = load_model_config(args.model_config, args.model_overrides)
    from training.distributed import init_distributed, place_mimi

    device = init_distributed()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(int(os.environ.get("POCKET_TTS_CPU_THREADS", max(1, (os.cpu_count() or 2) - 2))))  # pocket_tts pins 1 at import
    mimi = load_frozen_mimi(model_config)
    place_mimi(mimi, device)  # chunked streaming encode where MPS limits conv1d length
    if not args.data.train_jsonl:
        raise SystemExit("the config has no data.train_jsonl to precompute")
    precompute_manifest(
        Path(args.data.train_jsonl),
        mimi,
        device,
        batch_size,
        decode_workers,
        str(model_config.weights_path),
        stitch_cuts=stitch_cuts,
        max_voice_prompt_sec=args.data.max_voice_prompt_sec,
        max_duration_sec=args.data.max_duration_sec,
    )


if __name__ == "__main__":
    app()

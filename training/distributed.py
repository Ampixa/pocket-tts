import os

import torch
import torch.distributed as dist


def is_torchrun() -> bool:
    return "LOCAL_RANK" in os.environ


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def mps_conv1d_length_limited(device: torch.device) -> bool:
    """macOS 14 MPS rejects conv1d inputs longer than 65,536 samples (2.7 s at
    24 kHz; "Output channels > 65536 not supported"); every whole-clip Mimi
    encoder call is longer than that. Newer macOS has no such limit."""
    if device.type != "mps":
        return False
    try:
        torch.nn.functional.conv1d(
            torch.zeros(1, 1, 65537, device=device), torch.zeros(1, 1, 1, device=device)
        )
        return False
    except NotImplementedError:
        return True


def mimi_device_for(device: torch.device) -> torch.device:
    """Where the frozen Mimi codec runs: with `device`. Kept for the tests and
    for callers that only need the device; `place_mimi` does the placement."""
    return device


def place_mimi(mimi, device: torch.device) -> torch.device:
    """Move Mimi to `device`. Where MPS cannot run its encoder on whole clips,
    switch the encoder to chunked streaming encode (bit-for-bit the same
    computation, see training/modules/mimi_chunked.py) instead of falling back
    to CPU, which measured 37x slower."""
    mimi.to(device)
    if mps_conv1d_length_limited(device):
        from training.modules.mimi_chunked import enable_chunked_encode

        enable_chunked_encode(mimi)
    return device


def _require_cuda():
    """Training on CPU is accidental (a mismatched torch build), not a use case.

    Apple Silicon (MPS) is accepted as an accelerator: bf16 autocast and SDPA
    backward both work there from torch 2.5 on, at a fraction of CUDA speed.
    """
    if (
        torch.cuda.is_available()
        or mps_available()
        or os.environ.get("POCKET_TTS_ALLOW_CPU") == "1"
    ):
        return
    if torch.version.cuda is None:
        hint = (
            "this is a CPU-only torch build: pyproject pins the CPU wheel index for "
            "inference, so installing the train extra replaces a CUDA torch. Reinstall "
            "torch afterwards, see training/README.md."
        )
    else:
        hint = (
            "torch is built against CUDA "
            f"{torch.version.cuda}; if the driver reports an older version, install a "
            "matching build, see training/README.md."
        )
    raise SystemExit(
        f"no CUDA device visible to torch (torch {torch.__version__}).\n{hint}\n"
        "Set POCKET_TTS_ALLOW_CPU=1 to run on CPU anyway."
    )


def init_distributed() -> torch.device:
    _require_cuda()
    if is_torchrun():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank)
    forced = os.environ.get("POCKET_TTS_DEVICE")
    if forced:
        return torch.device(forced)  # cuda | mps | cpu; for A/B timing and debugging
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def shutdown_distributed():
    """Hold every rank until all are done, then tear NCCL down.

    Without the barrier, multi-GPU training can crash at the end.
    """
    if not dist.is_initialized():
        return
    dist.barrier(device_ids=[torch.cuda.current_device()])
    dist.destroy_process_group()


def avg_across_ranks(value: float) -> float:
    if not dist.is_initialized():
        return value
    t = torch.tensor(value, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()

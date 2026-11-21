import os
import re
from pathlib import Path

import torch

_NAME = re.compile(r"^ckpt-(\d{8})\.pt$")


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(directory: Path, step: int, state: dict, keep: int = 2) -> Path:
    """Write ckpt-<step>.pt so a reader sees either the whole file or none of it.

    The bytes go to a temporary file that is flushed to disk, then renamed. A kill
    at any point leaves the previous checkpoint intact.
    """
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"ckpt-{step:08d}.pt"
    tmp = final.with_suffix(".pt.tmp")
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)
    _fsync_dir(directory)
    for old in sorted(p for p in directory.iterdir() if _NAME.match(p.name))[:-keep]:
        old.unlink()
    for stale in directory.glob("*.tmp"):
        stale.unlink()
    return final


def load_latest(directory: Path):
    """Return (step, state) from the newest checkpoint that loads, or None.

    A checkpoint that cannot be read (cut short by a full disk, for example) is
    reported and skipped, and the one before it is used.
    """
    if not directory.exists():
        return None
    for p in sorted((p for p in directory.iterdir() if _NAME.match(p.name)), reverse=True):
        try:
            return int(_NAME.match(p.name).group(1)), torch.load(p, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001
            print(f"resumable: skipping unreadable checkpoint {p.name} ({type(exc).__name__}); trying the one before it", flush=True)
    return None

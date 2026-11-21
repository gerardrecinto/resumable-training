import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import torch
from torch import nn

from . import ckpt

PREEMPTED = 143  # 128 + SIGTERM, what Kubernetes and SLURM expect from a job that was asked to stop


def make_batch(seed: int, step: int, batch: int):
    """The batch for one step depends only on (seed, step), so a resumed run sees the same data."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + step)
    x = torch.randn(batch, 16, generator=g)
    y = torch.sin(x).sum(dim=1, keepdim=True)
    return x, y


def weights_sha256(model: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(bytes(v.detach().cpu().contiguous().flatten().view(torch.uint8).tolist()))
    return h.hexdigest()


def config_of(a) -> dict:
    return {"seed": a.seed, "lr": a.lr, "batch": a.batch, "amp": a.amp}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="resumable-train", description="Toy training loop with atomic checkpoints and resume.")
    ap.add_argument("--ckpt-dir", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast for the forward pass")
    a = ap.parse_args(argv)

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    step, resumed_from = 0, None

    found = ckpt.load_latest(a.ckpt_dir)
    if found:
        step, state = found
        if state["config"] != config_of(a):
            diff = ", ".join(f"{k}: {state['config'][k]} vs {v}" for k, v in config_of(a).items() if state["config"].get(k) != v)
            print(f"resumable: the checkpoint in {a.ckpt_dir} was made with different settings ({diff}). "
                  "Restore those settings or use a new --ckpt-dir.", file=sys.stderr)
            return 2
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        resumed_from = step
        print(f"resumable: resuming from step {step}", flush=True)

    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    crash_at = int(os.environ.get("RESUMABLE_CRASH_AT", "0"))  # test hook: die like a kill -9, with no cleanup
    slow = float(os.environ.get("RESUMABLE_SLOW", "0"))  # test hook: seconds to wait per step so a signal can land mid-run
    print("resumable: training started", flush=True)

    def snapshot():
        return {"model": model.state_dict(), "optimizer": opt.state_dict(), "config": config_of(a), "step": step}

    loss_value = float("nan")
    while step < a.steps:
        x, y = make_batch(a.seed, step, a.batch)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=a.amp):
            pred = model(x)
        loss = nn.functional.mse_loss(pred.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        loss_value = loss.item()
        if slow:
            time.sleep(slow)
        if crash_at and step == crash_at:
            os._exit(137)
        if step % a.ckpt_every == 0 or step == a.steps:
            ckpt.save(a.ckpt_dir, step, snapshot())
        if stop["now"]:
            ckpt.save(a.ckpt_dir, step, snapshot())
            print(f"resumable: SIGTERM at step {step}, checkpoint saved", flush=True)
            return PREEMPTED

    print(json.dumps({"step": step, "loss": round(loss_value, 6), "weights_sha256": weights_sha256(model), "resumed_from": resumed_from}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

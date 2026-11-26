# resumable-training

A small PyTorch training loop that survives being killed. A run that dies, or is told to stop, picks up from its last checkpoint and ends with exactly the same weights as a run that was never interrupted.

It runs on synthetic data. The point is the checkpoint and resume mechanics you would want in front of a real training job on spot nodes, Kubernetes or SLURM.

## Run it

Needs Python 3.10 or later. The first install downloads PyTorch.

```bash
git clone https://github.com/gerardrecinto/resumable-training && cd resumable-training
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or plain: pip install torch
pip install -e . pytest
python -m pytest -q
```

Kill a run and resume it (`RESUMABLE_CRASH_AT` makes the process die like `kill -9`):

```bash
resumable-train --ckpt-dir /tmp/a --steps 200 --ckpt-every 20
RESUMABLE_CRASH_AT=137 resumable-train --ckpt-dir /tmp/b --steps 200 --ckpt-every 20
resumable-train --ckpt-dir /tmp/b --steps 200 --ckpt-every 20
```

On my laptop the first and third commands printed the same line, apart from `resumed_from`:

```
{"step": 200, "loss": 0.861203, "weights_sha256": "cb675f14...dfc", "resumed_from": null}
{"step": 200, "loss": 0.861203, "weights_sha256": "cb675f14...dfc", "resumed_from": 120}
```

## What it does

- Writes each checkpoint to a temporary file, flushes it to disk and renames it, so a kill leaves the previous checkpoint whole. Keeps the newest two.
- Resumes from the newest checkpoint that loads. A cut-off file is named and skipped, and the one before it is used.
- On SIGTERM, which is what Kubernetes and SLURM send before they stop a job, saves a checkpoint at the next step and exits 143. Rerunning the same command continues.
- Refuses to resume if the settings (`seed`, `lr`, `batch`, `amp`) changed, and prints which one. Use a new `--ckpt-dir` to start over.
- `--amp` runs the forward pass under bfloat16 autocast. Resume equality is tested with it on and off.

Each step's data depends only on `(seed, step)`, so no data loader state has to be saved. If your job uses a shuffled loader or dropout, you also have to save and restore those random states, and this repo does not show that.

## What it does not do

One process on CPU. No multi-GPU, no sharded checkpoints, no learning-rate schedule, no mixed precision on GPU with a gradient scaler, and no upload to object storage. Exact equality was checked on one machine with one thread; a different PyTorch version or hardware can give different bits. Use it as the pattern, not as a framework.

To roll back, delete the checkpoint folder. Nothing outside it is written.

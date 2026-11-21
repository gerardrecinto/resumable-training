import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def run(ckpt_dir, *extra, env=None, steps=60):
    cmd = [sys.executable, "-m", "resumable.train", "--ckpt-dir", str(ckpt_dir), "--steps", str(steps), "--ckpt-every", "10", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=False, env={**os.environ, **(env or {})})


def final(proc):
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def straight(tmp_path_factory):
    p = run(tmp_path_factory.mktemp("straight"))
    assert p.returncode == 0, p.stderr
    return final(p)


@pytest.mark.parametrize("amp", [[], ["--amp"]])
def test_killed_run_resumes_to_the_same_weights(tmp_path, amp):
    base = final(run(tmp_path / "a", *amp))
    killed = run(tmp_path / "b", *amp, env={"RESUMABLE_CRASH_AT": "37"})
    assert killed.returncode == 137
    resumed = run(tmp_path / "b", *amp)
    assert "resuming from step 30" in resumed.stdout
    out = final(resumed)
    assert out["resumed_from"] == 30
    assert out["weights_sha256"] == base["weights_sha256"] and out["loss"] == base["loss"]


def test_a_cut_short_newest_checkpoint_falls_back_to_the_one_before(tmp_path, straight):
    run(tmp_path, env={"RESUMABLE_CRASH_AT": "45"})  # leaves ckpt-30 and ckpt-40
    newest = max(tmp_path.glob("ckpt-*.pt"))
    newest.write_bytes(newest.read_bytes()[:100])
    p = run(tmp_path)
    assert "skipping unreadable checkpoint ckpt-00000040.pt" in p.stdout and "resuming from step 30" in p.stdout
    assert final(p)["weights_sha256"] == straight["weights_sha256"]


def test_sigterm_saves_a_checkpoint_and_the_resumed_run_matches(tmp_path, straight):
    cmd = [sys.executable, "-m", "resumable.train", "--ckpt-dir", str(tmp_path), "--steps", "60", "--ckpt-every", "1000"]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env={**os.environ, "RESUMABLE_SLOW": "0.05"})
    for line in proc.stdout:  # wait until the signal handler is installed
        if "training started" in line:
            break
    time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 143 and "checkpoint saved" in out, (proc.returncode, out)
    saved = int(max(tmp_path.glob("ckpt-*.pt")).name[5:13])
    assert 0 < saved < 60
    resumed = final(run(tmp_path))
    assert resumed["resumed_from"] == saved and resumed["weights_sha256"] == straight["weights_sha256"]


def test_changed_settings_are_refused_with_the_difference(tmp_path):
    run(tmp_path, env={"RESUMABLE_CRASH_AT": "25"})
    p = run(tmp_path, "--lr", "0.5")
    assert p.returncode == 2 and "lr: 0.01 vs 0.5" in p.stderr and "--ckpt-dir" in p.stderr


def test_no_partial_file_is_left_behind_and_old_checkpoints_are_pruned(tmp_path):
    run(tmp_path)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["ckpt-00000050.pt", "ckpt-00000060.pt"]

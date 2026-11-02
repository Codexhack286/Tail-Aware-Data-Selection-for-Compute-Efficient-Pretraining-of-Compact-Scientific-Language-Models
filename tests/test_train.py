import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent


def make_fake_data(root: Path):
    data_dir = root / "data"
    run_dir = data_dir / "smoke"
    run_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rng.integers(0, 64, size=2000, dtype=np.uint16).tofile(run_dir / "train.bin")
    rng.integers(0, 64, size=500, dtype=np.uint16).tofile(data_dir / "val.bin")
    (data_dir / "meta.json").write_text(
        json.dumps({"vocab_size": 64, "eot_id": 0, "pad_id": 1})
    )
    return data_dir


def test_train_cli_writes_checkpoints_and_metrics(tmp_path):
    data_dir = make_fake_data(tmp_path)
    out_dir = tmp_path / "run"
    cmd = [
        sys.executable,
        "src/train.py",
        "--subset",
        "smoke",
        "--seed",
        "42",
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(out_dir),
        "--max-steps",
        "3",
        "--batch-size",
        "2",
        "--block-size",
        "8",
        "--n-layer",
        "1",
        "--n-head",
        "2",
        "--n-embd",
        "32",
        "--device",
        "cpu",
        "--no-amp",
        "--eval-every",
        "1",
        "--log-every",
        "1",
        "--save-every",
        "1",
    ]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert (out_dir / "ckpt_last.pt").exists()
    assert (out_dir / "model_last.pt").exists()
    assert (out_dir / "metrics.jsonl").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["step"] == 3
    assert summary["tokens_seen"] == 3 * 2 * 8
    assert summary["final_val_loss"] is not None

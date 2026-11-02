from pathlib import Path

from train_all import build_train_command, is_complete, should_resume


def test_build_train_command_uses_fixed_budget():
    cmd = build_train_command("random", 42, 60_000_000, python="python")
    assert cmd[:2] == ["python", "src/train.py"]
    assert "--subset" in cmd and "random" in cmd
    assert "--seed" in cmd and "42" in cmd
    assert "--max-tokens" in cmd and "60000000" in cmd


def test_should_resume_partial_run(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "ckpt_last.pt").write_bytes(b"x")
    assert should_resume(out) is True
    assert is_complete(out) is False


def test_is_complete_only_with_summary(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "summary.json").write_text("{}")
    assert is_complete(out) is True
    assert should_resume(out) is False

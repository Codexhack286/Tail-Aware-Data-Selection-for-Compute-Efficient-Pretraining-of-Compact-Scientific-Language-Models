# Part 3 Compact Transformer Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a 17.1M-parameter decoder-only Transformer on all five Part 2 subsets for 60M tokens each and two seeds, producing resumable checkpoints and comparable logs for Part 4.

**Architecture:** A GPT-style pre-LN Transformer implemented directly in PyTorch, fed by memmapped uint16 token files. A single `train.py` CLI owns one run; `train_all.py` runs ten independent runs sequentially; `plot_part3.py` aggregates the logs.

**Tech Stack:** Python 3.14, PyTorch 2.14.1+cu130, NumPy, Matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-11-02-part3-training-design.md`

## Global Constraints

- Do not read `test.bin` or use test performance for any decision in Part 3.
- Use the committed tokenizer and manifests; do not retrain either.
- Every subset uses the same model, optimizer, schedule, batch size, context length, token budget, and seed list.
- The token budget is exactly 60,000,000 tokens per run; implement it as a whole number of optimizer steps.
- Runs are resumable from `ckpt_last.pt` without changing the data order.
- Training must work on Windows with no DataLoader worker processes.
- No Unsloth, Transformers, or pretrained weights.

## Review Focus

- **Truncated or wrong token windows:** the target must be the input shifted by one token, including across document boundaries. Test: `test_windows_are_shifted_by_one`.
- **Causality leak:** changing a future token must not change earlier logits. Test: `test_future_tokens_do_not_change_past_logits`.
- **Budget drift between subsets:** two runs must use the same number of optimizer steps and tokens. Test: `test_steps_for_tokens` and `test_build_train_command_uses_fixed_budget`.
- **Resume changing the data order:** skipping to a resumed step must reproduce the same next batch. Test: `test_stream_skip_matches_advancing`.
- **Silent CPU fallback or mixed-precision failure:** the pilot must record the device, dtype, tokens/second, and peak memory. Verification: pilot summary contains all four.
- **A partial run being mistaken for a completed one:** `train_all.py` must resume when `ckpt_last.pt` exists but skip only when `summary.json` exists. Test: `test_should_resume_partial_run`.

---

### Task 1: Training dependency and missing Part 3 inputs

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `runs/.gitkeep` (ignored except the placeholder)
- Uses: committed `tokenizer/tokenizer.json`, `manifests/*.txt.gz`, `data/raw/source_info.json`

**Interfaces:**
- Produces: `data/processed/{train,val,test}.parquet`
- Produces: `data/selection/topics.parquet`
- Produces: `data/subsets/{random,quality,diversity,tail_aware,tail_only}/train.bin`
- Produces: `data/subsets/{val,test}.bin`, `data/subsets/{val,test}_docs.parquet`, `data/subsets/meta.json`
- Produces: a `.venv` with `torch 2.14.1+cu130` available through `uv run --group train`

- [ ] **Step 1: Add the train dependency group and CUDA index**

Append to `pyproject.toml`:

```toml
[dependency-groups]
train = ["torch==2.14.1"]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
```

Keep the existing `dev` group. Add `runs/` and `unsloth_compiled_cache/` to
`.gitignore`.

- [ ] **Step 2: Install and verify CUDA**

Run:

```bash
uv sync --group train
uv run --group train python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: a `2.14.1+cu130` version string, CUDA `13.0`, `True`, and
`NVIDIA GeForce RTX 5050 Laptop GPU`.

- [ ] **Step 3: Rebuild the missing Part 1/2 artifacts**

Run, in order (about 25 minutes):

```bash
uv run python src/build_raw.py
uv run python src/clean_corpus.py
uv run python src/split_corpus.py
uv run python src/build_topics.py
uv run python src/export_tokens.py
```

If the Kaggle `processedryan` parquet files are already available, place
`train.parquet`, `val.parquet`, and `test.parquet` in `data/processed/` and run
only the last two commands.

- [ ] **Step 4: Verify the handoff**

Run:

```bash
uv run python -c "import json, pathlib; m=json.loads(pathlib.Path('data/subsets/meta.json').read_text()); print(m['vocab_size'], m['eot_id'], m['pad_id']); print(m['files'])"
```

Expected: vocab `16384`, eot `0`, pad `1`; each subset train file is about
30,000,000 tokens; val and test are about 5.96M and 6.07M tokens.

Also verify dtype and token range once:

```bash
uv run python -c "import numpy as np; a=np.memmap('data/subsets/random/train.bin', dtype=np.uint16, mode='r'); print(len(a), int(a.min()), int(a.max()))"
```

Expected: length near 30M, minimum 0, maximum below 16384.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore
git commit -m "build: add Part 3 training dependency and ignore run artifacts"
```

---

### Task 2: GPT model

**Files:**
- Create: `src/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `GPTConfig(vocab_size, block_size, n_layer, n_head, n_embd, dropout, bias)` frozen dataclass with defaults 16384, 512, 6, 6, 384, 0.0, True.
- Produces: `GPT(config: GPTConfig)` with `forward(idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]` and `num_parameters() -> int`.
- Produces: `GPT.lm_head` and `GPT.tok_emb` sharing one weight tensor.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model.py`:

```python
import pytest

torch = pytest.importorskip("torch")

from model import GPT, GPTConfig


def tiny_config():
    return GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)


def test_forward_shapes_and_loss():
    model = GPT(tiny_config())
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 64)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_future_tokens_do_not_change_past_logits():
    torch.manual_seed(0)
    model = GPT(tiny_config()).eval()
    x = torch.randint(0, 64, (2, 16))
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % 64
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(x2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_default_config_is_about_17m_params_and_tied():
    model = GPT(GPTConfig())
    assert 17_000_000 < model.num_parameters() < 17_300_000
    assert model.lm_head.weight is model.tok_emb.weight
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --group train pytest tests/test_model.py -v`

Expected: collection error or `ModuleNotFoundError: No module named 'model'`.

- [ ] **Step 3: Implement `src/model.py`**

Implement the GPT with these exact components:

- `CausalSelfAttention`: one `c_attn` linear to 3x width, `c_proj`, reshape to
  `(B, n_head, T, head_dim)`, and
  `F.scaled_dot_product_attention(q, k, v, dropout_p=..., is_causal=True)`.
- `MLP`: `c_fc` to 4x width, GELU, `c_proj`.
- `Block`: pre-LayerNorm attention and MLP with residual connections.
- `GPT`: token embedding, learned position embedding, dropout, blocks, final
  LayerNorm, and a bias-free `lm_head` tied to `tok_emb.weight`.
- Initialization: normal(0, 0.02) for weights, zeros for biases, and residual
  projection weights scaled by `1 / sqrt(2 * n_layer)`.
- `forward`: compute logits; if `targets` is provided, compute mean
  cross-entropy over all positions and return `(logits, loss)`.
- `num_parameters`: count unique parameters (the tied weight counts once).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --group train pytest tests/test_model.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: add compact GPT model for Part 3"
```

---

### Task 3: Token window data pipeline

**Files:**
- Create: `src/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `TokenWindows(path, block_size)` with `.data` (uint16 memmap), `.n_windows`, `.get_batch(indices) -> (Tensor, Tensor)`, and `.val_batches(batch_size) -> Iterator[(Tensor, Tensor)]`.
- Produces: `TokenStream(path, block_size, batch_size, grad_accum, seed)` with `.next_micro_batch()` and `.skip_steps(steps)`.
- Produces: `steps_for_tokens(max_tokens, block_size, batch_size, grad_accum) -> int`.
- Windows use `block_size` input tokens and `block_size` target tokens shifted by one. Training windows are non-overlapping and shuffled once per epoch.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data.py`:

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from data import TokenStream, TokenWindows, steps_for_tokens


def write_bin(tmp_path, n=1000):
    path = tmp_path / "train.bin"
    np.arange(n, dtype=np.uint16).tofile(path)
    return path


def test_windows_are_shifted_by_one(tmp_path):
    windows = TokenWindows(write_bin(tmp_path), block_size=8)
    x, y = windows.get_batch(np.array([0, 1]))
    assert x.tolist()[0] == list(range(8))
    assert y.tolist()[0] == list(range(1, 9))
    assert x.tolist()[1] == list(range(8, 16))
    assert y.tolist()[1] == list(range(9, 17))


def test_validation_batches_cover_all_windows(tmp_path):
    windows = TokenWindows(write_bin(tmp_path), block_size=8)
    batches = list(windows.val_batches(batch_size=32))
    assert sum(x.shape[0] for x, _ in batches) == windows.n_windows


def test_steps_for_tokens():
    assert steps_for_tokens(1000, 8, 2, 2) == 31


def test_stream_skip_matches_advancing(tmp_path):
    path = write_bin(tmp_path)
    a = TokenStream(path, block_size=8, batch_size=2, grad_accum=1, seed=7)
    for _ in range(5):
        a.next_micro_batch()
    expected = a.next_micro_batch()
    b = TokenStream(path, block_size=8, batch_size=2, grad_accum=1, seed=7)
    b.skip_steps(5)
    actual = b.next_micro_batch()
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --group train pytest tests/test_data.py -v`

Expected: `ModuleNotFoundError: No module named 'data'`.

- [ ] **Step 3: Implement `src/data.py`**

Implementation details:

- `n_windows = (len(data) - 1) // block_size`.
- `get_batch(indices)` reads `data[i*block:(i+1)*block]` and
  `data[i*block+1:(i+1)*block+1]`, returning `torch.long` tensors.
- `val_batches` iterates sequentially over all window indices and yields a
  final partial batch rather than dropping it.
- `TokenStream` shuffles the window indices with
  `np.random.default_rng(seed).permutation(n_windows)` at the start of each
  epoch, drops a remainder smaller than `batch_size`, and advances to the next
  epoch when needed.
- `skip_steps(steps)` advances `steps * grad_accum` micro-batches without
  materializing tensors, so resume is cheap and deterministic.
- `steps_for_tokens` is integer floor division of the token budget by
  `block_size * batch_size * grad_accum`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --group train pytest tests/test_data.py -v`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat: add memmapped token window pipeline"
```

---

### Task 4: Training loop and CLI

**Files:**
- Create: `src/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Produces: `TrainConfig` dataclass with the defaults in the spec.
- Produces: `train(config: TrainConfig) -> dict` returning the summary dictionary.
- Produces: `main(argv: list[str] | None = None) -> None`.
- CLI flags: `--subset`, `--seed`, `--max-tokens`, `--block-size`,
  `--batch-size`, `--grad-accum`, `--lr`, `--min-lr-ratio`, `--warmup-frac`,
  `--weight-decay`, `--grad-clip`, `--dropout`, `--n-layer`, `--n-head`,
  `--n-embd`, `--eval-every`, `--log-every`, `--save-every`, `--device`,
  `--no-amp`, `--max-steps`, `--data-dir`, `--out-dir`, `--resume`,
  `--dry-run`.
- Consumes: `GPT`/`GPTConfig` from `model.py`; `TokenWindows`/`TokenStream`/
  `steps_for_tokens` from `data.py`; `config.SUBSETS_DIR` and
  `config.RUNS_DIR`.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_train.py`:

```python
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
    (data_dir / "meta.json").write_text(json.dumps({"vocab_size": 64, "eot_id": 0, "pad_id": 1}))
    return data_dir


def test_train_cli_writes_checkpoints_and_metrics(tmp_path):
    data_dir = make_fake_data(tmp_path)
    out_dir = tmp_path / "run"
    cmd = [
        sys.executable, "src/train.py",
        "--subset", "smoke",
        "--seed", "42",
        "--data-dir", str(data_dir),
        "--out-dir", str(out_dir),
        "--max-steps", "3",
        "--batch-size", "2",
        "--block-size", "8",
        "--n-layer", "1",
        "--n-head", "2",
        "--n-embd", "32",
        "--device", "cpu",
        "--no-amp",
        "--eval-every", "1",
        "--log-every", "1",
        "--save-every", "1",
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group train pytest tests/test_train.py -v`

Expected: subprocess fails because `src/train.py` does not exist.

- [ ] **Step 3: Implement `src/train.py`**

Implementation requirements:

- Resolve the device (`auto` uses CUDA when available) and print/save it.
- Load vocab size and special ids from `data_dir/meta.json`; default to
  `config.VOCAB_SIZE` and eot id 0 if the file is absent.
- Compute `max_steps = steps_for_tokens(...)` unless `--max-steps` is given.
- Build the model, AdamW param groups (decay for weights with ndim >= 2, no
  decay for biases and LayerNorm), and a LambdaLR with linear warmup to peak
  then cosine decay to `min_lr_ratio * lr`.
- AdamW uses `fused=True` only on CUDA.
- Use bf16 autocast on CUDA unless `--no-amp`; no autocast on CPU.
- Each optimizer step accumulates `grad_accum` micro-batches, clips gradients
  at `grad_clip`, steps the optimizer and scheduler, and records the mean
  training loss.
- Log every `log_every` steps to `metrics.jsonl` with step, tokens, train loss,
  learning rate, gradient norm, elapsed seconds, tokens/second, and peak
  allocated GPU MB.
- Evaluate the full validation set every `eval_every` steps and at the end,
  recording `val_loss`, `val_ppl`, and `best_val`.
- Save `ckpt_last.pt` (model, optimizer, scheduler, step, config) and
  `ckpt_best.pt` (model only, on validation improvement) every `save_every`
  steps and at the end. Save `model_last.pt` at the end.
- `--resume` loads `ckpt_last.pt`, restores model/optimizer/scheduler, and
  calls `TokenStream.skip_steps(start_step)`.
- Write `config.json`, `summary.json`, and `run.log` in `out_dir`.
- `--dry-run` prints the resolved configuration and step count without
  training.
- Guard `test.bin`: do not open it anywhere.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group train pytest tests/test_train.py -v`

Expected: 1 passed, and the subprocess exits 0.

- [ ] **Step 5: Run the full suite**

Run: `uv run --group train pytest -q`

Expected: all existing Part 1/2 tests plus the new tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: add resumable Part 3 training loop"
```

---

### Task 5: Batch runner and report aggregation

**Files:**
- Create: `src/train_all.py`
- Create: `src/plot_part3.py`
- Test: `tests/test_train_all.py`
- Test: `tests/test_plot_part3.py`

**Interfaces:**
- Produces: `build_train_command(subset, seed, max_tokens, python=None, extra=()) -> list[str]`.
- Produces: `should_resume(out_dir: Path) -> bool` and
  `is_complete(out_dir: Path) -> bool`.
- Produces: `collect_runs(runs_dir: Path) -> list[dict]` and
  `summarize(runs: list[dict]) -> dict`.
- Produces: `write_report(runs_dir, out_path)` and
  `plot_val_curves(runs, out_path)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_all.py`:

```python
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
```

Create `tests/test_plot_part3.py`:

```python
import json
from pathlib import Path

from plot_part3 import collect_runs, summarize


def test_collect_and_summarize(tmp_path: Path):
    run = tmp_path / "random" / "seed42"
    run.mkdir(parents=True)
    rows = [
        {"step": 1, "tokens": 16384, "train_loss": 4.0, "val_loss": 3.8, "val_ppl": 44.7},
        {"step": 2, "tokens": 32768, "train_loss": 3.5, "val_loss": 3.4, "val_ppl": 30.0},
    ]
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (run / "summary.json").write_text(json.dumps({
        "subset": "random", "seed": 42, "final_val_loss": 3.4,
        "final_val_ppl": 30.0, "best_val_loss": 3.4, "tokens_seen": 32768,
        "tokens_per_s": 10000.0, "wall_time_s": 3.3, "peak_mem_mb": 512.0,
    }))
    runs = collect_runs(tmp_path)
    assert len(runs) == 1
    report = summarize(runs)
    assert report["random"]["runs"] == 1
    assert report["random"]["mean_final_val_loss"] == 3.4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --group train pytest tests/test_train_all.py tests/test_plot_part3.py -v`

Expected: `ModuleNotFoundError: No module named 'train_all'` and
`No module named 'plot_part3'`.

- [ ] **Step 3: Implement `src/train_all.py`**

Requirements:

- `SUBSETS = ("random", "quality", "diversity", "tail_aware", "tail_only")`.
- CLI: `--seeds` (nargs+, default 42 43), `--subsets` (nargs+, default all),
  `--max-tokens` (default 60,000,000), `--force`, `--dry-run`, `--python`.
- Build commands with the chosen Python, `src/train.py`, subset, seed, and
  max tokens.
- For each run: skip if `summary.json` exists and not `--force`; otherwise add
  `--resume` if `ckpt_last.pt` exists; print the command; run it with
  `subprocess.run(..., cwd=ROOT, check=True)`.
- Write `runs/schedule.json` before starting, containing every command, and
  update it with status (`pending`, `running`, `complete`, `failed`) around
  each run.
- Stop on the first failure and print the failed command.

- [ ] **Step 4: Implement `src/plot_part3.py`**

Requirements:

- `collect_runs` walks `runs/<subset>/seed<seed>/metrics.jsonl`, attaches the
  matching `summary.json` if present, and returns a sorted list.
- `summarize` groups by subset and reports the number of runs, mean and sample
  standard deviation of final validation loss, mean final perplexity, mean
  tokens/second, mean wall-clock seconds, and mean peak memory.
- `write_report` writes the summary as indented JSON.
- `plot_val_curves` plots validation loss against tokens seen for every run,
  one color per subset, and writes a PNG with the Matplotlib Agg backend.
- CLI: `--runs-dir` (default `runs`), `--out-json` (default
  `reports/part3_report.json`), `--out-fig` (default
  `reports/figures/part3_val_loss.png`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --group train pytest tests/test_train_all.py tests/test_plot_part3.py -v`

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/train_all.py src/plot_part3.py tests/test_train_all.py tests/test_plot_part3.py
git commit -m "feat: add Part 3 batch runner and report aggregation"
```

---

### Task 6: Pilot, documentation, and handoff

**Files:**
- Create: `PART3.md`
- Modify: `README.md`
- Modify: `src/config.py` (add `RUNS_DIR` and Part 3 defaults)
- Generated: `reports/part3_report.json`, `reports/figures/part3_val_loss.png` after full training

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: a pilot summary with device, dtype, tokens/second, and peak memory.
- Produces: the exact full-training command for the project owner.

- [ ] **Step 1: Add Part 3 defaults to `src/config.py`**

Add:

```python
# Part 3: compact Transformer training
RUNS_DIR = ROOT / "runs"
PART3_BLOCK_SIZE = 512
PART3_N_LAYER = 6
PART3_N_HEAD = 6
PART3_N_EMBD = 384
PART3_DROPOUT = 0.0
PART3_MAX_TOKENS = 60_000_000
PART3_BATCH_SIZE = 32
PART3_GRAD_ACCUM = 1
PART3_LR = 6e-4
PART3_MIN_LR_RATIO = 0.1
PART3_WARMUP_FRAC = 0.02
PART3_WEIGHT_DECAY = 0.1
PART3_GRAD_CLIP = 1.0
PART3_EVAL_EVERY = 250
PART3_LOG_EVERY = 10
PART3_SAVE_EVERY = 500
PART3_SEEDS = (42, 43)
```

Use these as CLI defaults in `train.py` and `train_all.py`.

- [ ] **Step 2: Run the pilot**

Run:

```bash
uv run --group train python src/train.py --subset random --seed 42 --max-steps 200 --out-dir runs/pilot/random_seed42 --eval-every 100
```

Expected: 200 steps complete, `runs/pilot/random_seed42/summary.json` exists,
and the summary contains `device`, `amp_dtype`, `tokens_per_s`, and
`peak_mem_mb`. The pilot run is not part of the final comparison and can be
deleted.

- [ ] **Step 3: Record the pilot result and choose the final batch size**

If peak memory is below ~6 GB, keep batch size 32. If it is above, use gradient
accumulation 2 with batch size 16 (same 16,384 tokens per optimizer step). Do
not change the model width unless the pilot throughput makes the full ten-run
schedule exceed the owner's time budget. Any change must apply to all runs.

- [ ] **Step 4: Write `PART3.md`**

Include: goal, inputs, model configuration, training configuration, exact run
order, pilot throughput, full command, output layout, resume command, and the
caveat that overall validation loss is not the final test and that Part 4 must
report per-group results. Leave a clearly marked "Results" section to fill
after the full training.

- [ ] **Step 5: Update `README.md`**

Add a Part 3 section with the environment setup, pilot command, full command,
report command, and a pointer to `PART3.md`.

- [ ] **Step 6: Run the full test suite and verify the handoff**

Run:

```bash
uv run --group train pytest -q
uv run --group train python src/train_all.py --dry-run --seeds 42 43 --max-tokens 60000000
```

Expected: all tests pass and the dry run prints ten commands, one per
subset/seed pair.

- [ ] **Step 7: Commit**

```bash
git add src/config.py PART3.md README.md
git commit -m "docs: add Part 3 training guide and defaults"
```

---

## Self-Review

- Spec coverage: environment, data, model, optimization, checkpoints, two
  seeds, 60M tokens, no test-set use, and the handoff command all have a task.
- Type consistency: `GPTConfig`/`GPT`, `TokenWindows`/`TokenStream`,
  `steps_for_tokens`, `TrainConfig`/`train`, `build_train_command`,
  `collect_runs`/`summarize` are named consistently across tasks.
- Review Focus: each line names a test or a pilot check in the owning task.
- Proportion: the plan is long because the training loop is the new subsystem;
  implementation files stay focused.

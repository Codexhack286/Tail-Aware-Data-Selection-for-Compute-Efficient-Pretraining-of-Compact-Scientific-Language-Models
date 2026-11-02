# Part 3 Design: Compact Transformer Training

**Date:** 2026-11-02
**Status:** Approved by the project owner
**Scope:** Part 3 only (training and training logs); final test-set evaluation is Part 4.

## Intent

Train the same compact decoder-only Transformer on each of the five Part 2
training subsets, with the same tokenizer, architecture, optimization setup,
and training-token budget. The output is a set of checkpoints and comparable
training logs that Part 4 can evaluate without retraining.

The comparison is deliberately a controlled experiment: only the `train.bin`
file changes between runs. The code, random seed, model initialization,
optimizer, schedule, and validation set are held constant.

## Goals

- Train `random`, `quality`, `diversity`, `tail_aware`, and `tail_only` for
  exactly the same number of tokens.
- Use two training seeds per subset (42 and 43), giving ten runs.
- Produce resumable checkpoints, JSONL metrics, and a machine-readable summary
  for every run.
- Keep the test split untouched until Part 4.
- Provide a single terminal command the project owner can monitor.

## Non-goals

- No test-set evaluation, topic-wise test perplexity, or generation study.
  Those belong to Part 4.
- No instruction tuning, LoRA, or pretrained-model finetuning.
- No changes to the Part 1 or Part 2 data-selection logic.
- No Unsloth dependency. Unsloth is a finetuning accelerator; this is a
  from-scratch 17M-parameter model. Plain PyTorch is the correct tool.

## Decisions

### Environment

- Use the project `.venv` (Python 3.14.6) with a `train` dependency group.
- Install `torch==2.14.1` from the CUDA 13.0 wheel index. The RTX 5050 Laptop
  is compute capability 12.0 with 8.5 GB VRAM and supports bf16.
- Keep data preparation dependencies (Part 1/2) in the base environment; do not
  make a data-only install pull PyTorch.

### Data

- Use the committed tokenizer (`tokenizer/tokenizer.json`) and the committed
  manifests. Do not retrain or reselect anything.
- Rebuild only the missing local artifacts: processed splits, topic labels, and
  tokenized `.bin` files.
- `train.bin` for each subset is 30M uint16 tokens. `val.bin` is the shared
  full validation split (5.96M tokens). `test.bin` is created but not read
  during Part 3.
- Training windows are non-overlapping blocks of `block_size + 1` tokens,
  shuffled once per epoch. This gives deterministic coverage and avoids the
  duplicate-token variance of sampling with replacement.
- The token budget is **60M tokens per run** (two passes over each 30M-token
  subset). Every subset gets the same number of optimizer steps.
- Documents already contain one `<|endoftext|>` token at the end; no extra
  special handling is required.

### Model

- GPT-style decoder-only Transformer with pre-LayerNorm blocks, learned
  absolute position embeddings, GELU MLPs, causal scaled dot-product attention,
  and tied input/output embeddings.
- Default configuration: vocabulary 16,384; context 512; 6 layers; 6 heads;
  embedding width 384; dropout 0.0. Approximately 17.1M parameters.
- `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)` is
  used so the CUDA flash backend can handle the mask.
- A 17.1M model is near the top of the synopsis's 10-20M range. The pilot may
  reduce width to 320 (about 12.8M) if throughput is worse than expected, but
  the chosen configuration must be identical for all ten runs.

### Optimization

- AdamW, betas (0.9, 0.95), weight decay 0.1, no weight decay on biases and
  LayerNorm parameters.
- Peak learning rate 6e-4, 2% linear warmup, cosine decay to 10% of peak.
- Gradient clipping at 1.0.
- Batch size 32 sequences x 512 tokens = 16,384 tokens per optimizer step;
  gradient accumulation 1 unless the pilot shows an 8 GB memory problem.
- bf16 autocast on CUDA; fp32 fallback on CPU.
- Same seed for model initialization and data order in every subset. With two
  seeds, each seed changes both initialization and order, which is the correct
  unit for an error bar.

### Checkpoints and logs

Each run writes to `runs/<subset>/seed<seed>/`:

- `config.json` — resolved model, data, and training configuration.
- `metrics.jsonl` — one JSON object per log/eval step.
- `run.log` — human-readable progress log.
- `ckpt_last.pt` — model, optimizer, scheduler, and step for resuming.
- `ckpt_best.pt` — best-validation checkpoint (model only).
- `model_last.pt` — final model only, convenient for Part 4.
- `summary.json` — final/best validation loss, perplexity, tokens/second,
  wall-clock time, and peak GPU memory.

The primary comparison uses the final checkpoint at the fixed token budget.
`ckpt_best.pt` is for analysis only; choosing different early-stopping points
would make the comparison unfair.

### Validation policy

- `val.bin` is used for monitoring and checkpoint selection only.
- `test.bin` is never read in Part 3.
- Overall validation loss is reported, but it is not a clean test of tail
  coverage: `tail_aware` deliberately halves the head. Part 4 must report
  head/mid/tail perplexity separately.

## Handoff

The project owner runs the full training in a separate terminal so they can
monitor progress. The complete command is:

```bash
uv run --group train python src/train_all.py --seeds 42 43 --max-tokens 60000000
```

`train_all.py` skips completed runs, resumes interrupted runs, and stops on the
first failure. After all runs, `src/plot_part3.py` produces the comparison
report and validation-loss figure.

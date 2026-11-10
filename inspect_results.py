"""Read-only extraction of concrete examples from reports/deep_analysis.json."""
import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
data = json.load(open("reports/deep_analysis.json", encoding="utf-8"))
records = data["records"]

print("records:", len(records))
print("settings:", json.dumps(data["settings"], indent=2))

print("\n" + "=" * 80)
print("BOUNDARY / DECODE CONSISTENCY")
print("=" * 80)
for suffix in data["settings"]["prompt_suffixes"]:
    rows = [r for r in records if r["prompt_suffix"] == suffix]
    print(
        f"{suffix:7} boundary_joined={sum(r['boundary_joined'] for r in rows)}/{len(rows)} "
        f"full==prompt+cont={sum(r['prompt_plus_continuation_equals_full_skip'] for r in rows)}/{len(rows)}"
    )
joined = [r for r in records if r["boundary_joined"]]
print("joined suffixes:", Counter(r["prompt_suffix"] for r in joined))
print("replacement char in skip-decoded full text:",
      sum("\ufffd" in r["full_text_skip_special"] for r in records))
print("literal '<|endoftext|>' in skip-decoded full text:",
      sum("<|endoftext|>" in r["full_text_skip_special"] for r in records))

print("\n" + "=" * 80)
print("SAME MODEL / PROMPT / SEED: eot2 vs none")
print("=" * 80)
target = next(
    r for r in records
    if r["model"] == "tail_aware" and r["seed"] == 42 and r["prompt_index"] == 0
    and r["sample_seed"] == 0 and r["prompt_suffix"] == "eot2"
)
for r in records:
    if (r["model"], r["seed"], r["prompt_index"], r["sample_seed"]) != (
        target["model"], target["seed"], target["prompt_index"], target["sample_seed"]
    ):
        continue
    print(f"\n--- suffix={r['prompt_suffix']} ---")
    print("prompt_text           :", repr(r["prompt_text"]))
    print("prompt_suffix_ids      :", r["prompt_suffix_token_ids"])
    print("first generated pieces :", r["first_32_generated_pieces"][:12])
    print("continuation (first 220):", repr(r["continuation_text"][:220]))
    print("full skip (first 220)  :", repr(r["full_text_skip_special"][:220]))
    print("full keep (first 220)  :", repr(r["full_text_keep_special"][:220]))
    print("boundary_joined        :", r["boundary_joined"])
    print("stop_reason            :", r["stop_reason"], "| eos_position:", r["eos_position"])

print("\n" + "=" * 80)
print("WORST REPETITION CASE (by 4-gram max count)")
print("=" * 80)
worst = max(records, key=lambda r: r["repetition"]["token_4gram"]["max_count"])
print("model/suffix/seed:", worst["model"], worst["prompt_suffix"], worst["seed"], "sample", worst["sample_seed"])
print("prompt:", worst["prompt"])
print("4gram stats:", json.dumps(worst["repetition"]["token_4gram"], indent=2))
print("3gram stats:", json.dumps(worst["repetition"]["token_3gram"], indent=2))
print("longest same-token run:", worst["repetition"]["longest_same_token_run"])
print("continuation:\n", worst["continuation_text"][:1200])

print("\n" + "=" * 80)
print("TOKENIZATION IS LOSSLESS ACROSS GENERATED PIECES")
print("=" * 80)
ex = next(r for r in records if r["prompt_suffix"] == "none"
          and r["model"] == "quality" and r["prompt_index"] == 0 and r["sample_seed"] == 0)
pieces = ex["first_32_generated_pieces"]
ids = ex["generated_token_ids"][:len(pieces)]
print("ids   :", ids[:16])
print("pieces:", pieces[:16])
print("joined:", repr("".join(pieces[:16])))
print("text  :", repr(ex["continuation_text"][:120]))
print("joined-pieces prefix matches continuation:",
      "".join(pieces) == ex["continuation_text"][:len("".join(pieces))] if all(p is not None for p in pieces) else None)

print("\n" + "=" * 80)
print("STOP-REASON BREAKDOWN")
print("=" * 80)
for suffix in data["settings"]["prompt_suffixes"]:
    rows = [r for r in records if r["prompt_suffix"] == suffix]
    print(suffix, Counter(r["stop_reason"] for r in rows))

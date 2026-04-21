"""Load CommitBench, filter to clean conventional-commit examples, split, save as JSONL.

Order of operations (cheap -> expensive):
  1. Load full dataset.
  2. Cheap string filters: lang allow-list, conventional-commit regex,
     message length 5-200, diff length under MAX_DIFF_CHARS.
  3. Tokenize survivors with Qwen2.5-Coder tokenizer, drop those over
     MAX_DIFF_TOKENS.
  4. Shuffle (seed=42), split 15K/500/500, write JSONL.
  5. Print 10 random training examples for eyeball sanity.

Outputs:
  data/train.jsonl
  data/eval.jsonl
  data/test.jsonl

Each line: {"diff": str, "message": str}
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET_NAME = "Maxscha/commitbench"
TOKENIZER_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

# CommitBench's real schema: language field is `diff_languages` and values
# are short codes ("py", "js", "go", "java"). Multi-file diffs use
# comma-joined codes (e.g. "py,py", "js,js,json").
LANG_FIELD = "diff_languages"
ALLOWED_LANG_CODES = {"py", "js", "ts", "go", "java"}

# Conventional-commit header. Permits optional scope and optional `!`
# breaking-change marker. Only matches the FIRST line of the message.
#   feat: x                  yes
#   feat(scope): x           yes
#   feat(scope)!: x          yes
#   feat!: x                 yes
#   wip: x                   no  (not in type allow-list)
CONVENTIONAL_COMMIT_RE = re.compile(
    r"^(feat|fix|docs|style|refactor|test|chore)(\(.+?\))?!?: .+"
)

MIN_MSG_CHARS = 5
MAX_MSG_CHARS = 200
MAX_DIFF_CHARS = 5000
MAX_DIFF_TOKENS = 1500
# CommitBench anonymizes numeric literals in messages as `<I>` (24.7% of
# rows in a 1000-row sample). Training on these would teach the model to
# emit the placeholder at inference time. Reject.
REJECT_MSG_TOKENS = ("<I>",)
TARGET_TRAIN = 15_000
TARGET_EVAL = 500
TARGET_TEST = 500
SEED = 42

OUT_DIR = Path(__file__).parent


def cheap_filter(row: dict) -> bool:
    msg = row.get("message")
    diff = row.get("diff")
    if not msg or not diff:
        return False

    # Multi-file diffs come as comma-joined codes ("py,py,json"). Require
    # every token to be in the allow-list, so "py,py" passes but "py,php"
    # doesn't - we don't want training rows with disallowed-language files.
    lang_field = str(row.get(LANG_FIELD, "")).strip().lower()
    if not lang_field:
        return False
    lang_tokens = [t.strip() for t in lang_field.split(",") if t.strip()]
    if not all(t in ALLOWED_LANG_CODES for t in lang_tokens):
        return False

    if not (MIN_MSG_CHARS <= len(msg) <= MAX_MSG_CHARS):
        return False

    if len(diff) > MAX_DIFF_CHARS:
        return False

    if any(tok in msg for tok in REJECT_MSG_TOKENS):
        return False

    first_line = msg.strip().splitlines()[0] if msg.strip() else ""
    if not CONVENTIONAL_COMMIT_RE.match(first_line):
        return False

    return True


def token_count(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def main() -> None:
    print(f"Loading {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"  raw rows: {len(ds):,}")

    print("Applying cheap filters...")
    survivors = [row for row in ds if cheap_filter(row)]
    print(f"  after cheap filter: {len(survivors):,}")

    print(f"Loading tokenizer {TOKENIZER_NAME}...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print("Applying exact token-length filter...")
    survivors = [
        row for row in survivors if token_count(row["diff"], tok) <= MAX_DIFF_TOKENS
    ]
    print(f"  after token filter: {len(survivors):,}")

    rng = random.Random(SEED)
    rng.shuffle(survivors)

    needed = TARGET_TRAIN + TARGET_EVAL + TARGET_TEST
    if len(survivors) < needed:
        print(f"  WARNING: only {len(survivors):,} survivors, wanted {needed:,}")
    survivors = survivors[:needed]

    train = survivors[:TARGET_TRAIN]
    eval_ = survivors[TARGET_TRAIN : TARGET_TRAIN + TARGET_EVAL]
    test = survivors[TARGET_TRAIN + TARGET_EVAL :]

    for name, split in (("train", train), ("eval", eval_), ("test", test)):
        out_path = OUT_DIR / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in split:
                record = {"diff": row["diff"], "message": row["message"]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  wrote {len(split):,} rows -> {out_path.name}")

    print("\n=== Eyeball sanity - 10 random training examples ===")
    for row in rng.sample(train, min(10, len(train))):
        print("-" * 80)
        print("MSG:  ", row["message"][:120])
        print("DIFF: ", row["diff"][:200].replace("\n", "\\n"))


if __name__ == "__main__":
    main()

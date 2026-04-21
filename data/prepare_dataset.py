"""Load CommitBench, filter to clean conventional-commit examples, split, save as JSONL.

Written to run in Google Colab (where the Qwen tokenizer + datasets live). Locally
this script will only run up to the tokenizer download step before Colab-specific
concerns kick in.

Outputs:
    data/train.jsonl
    data/eval.jsonl
    data/test.jsonl

Each line: {"diff": str, "message": str}

TODO(Day 1): implement after verifying CommitBench schema.
"""

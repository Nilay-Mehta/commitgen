"""Evaluate commitgen on the test split.

Loads base Qwen2.5-Coder-0.5B-Instruct, optionally stacks a LoRA adapter,
generates commit messages for each (diff, message) pair in a test JSONL,
and reports:
  - corpus BLEU-4 (sacrebleu)
  - mean ROUGE-L F1 (rouge-score)
  - type accuracy: % of predictions whose conventional-commit type
    (feat/fix/docs/style/refactor/test/chore) matches the reference
  - length ratio: mean(len(pred)) / mean(len(ref))

Designed to run twice - once with `--adapter ...` for the tuned model and
once without for the base zero-shot baseline - so the two metrics JSON
files can be diffed.

CLI:
    python evaluation/evaluate.py \\
        --test-jsonl data/test.jsonl \\
        --output-dir evaluation/results \\
        --label tuned \\
        --adapter /content/drive/MyDrive/commitgen_checkpoints/qwen25coder-lora-full-0/checkpoint-2000

Outputs (per label):
    {output-dir}/metrics_{label}.json
    {output-dir}/predictions_{label}.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

import sacrebleu
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
SYSTEM_PROMPT = (
    "You are a helpful assistant that writes concise conventional-commit "
    "messages from git diffs. Output only the commit message, no extra "
    "commentary."
)
USER_TEMPLATE = "Write a conventional-commit message for this diff:\n\n{diff}"
# Same allow-list of commit types used during dataset filtering. Predictions
# outside this set count as type-mismatch.
COMMIT_TYPE_RE = re.compile(
    r"^(feat|fix|docs|style|refactor|test|chore)(\(.+?\))?!?: "
)
MAX_NEW_TOKENS = 64


def load_model(adapter_path: Optional[Path], device: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # bf16 helps on Ampere+, but on CPU and on Turing (T4) it's actually
    # slower than fp32. Pick per device.
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="eager",
    )

    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return model, tokenizer


def build_prompt(diff: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(diff=diff)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_one(model, tokenizer, diff: str) -> str:
    prompt = build_prompt(diff, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs.input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Commit messages are one line; if the model rambled, keep only the first.
    return text.splitlines()[0] if text else ""


def extract_type(message: str) -> Optional[str]:
    m = COMMIT_TYPE_RE.match(message)
    return m.group(1) if m else None


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_f = [
        scorer.score(r, p)["rougeL"].fmeasure
        for p, r in zip(predictions, references)
    ]
    rouge_l = sum(rouge_l_f) / len(rouge_l_f) * 100

    pred_types = [extract_type(p) for p in predictions]
    ref_types = [extract_type(r) for r in references]
    type_matches = sum(
        1 for p, r in zip(pred_types, ref_types) if p == r and p is not None
    )
    type_acc = type_matches / len(predictions) * 100

    avg_pred_len = sum(len(p) for p in predictions) / len(predictions)
    avg_ref_len = sum(len(r) for r in references) / len(references)

    return {
        "bleu4": round(bleu, 2),
        "rouge_l": round(rouge_l, 2),
        "type_accuracy_pct": round(type_acc, 2),
        "length_ratio": round(avg_pred_len / avg_ref_len, 3),
        "avg_pred_length": round(avg_pred_len, 1),
        "avg_ref_length": round(avg_ref_len, 1),
        "n_samples": len(predictions),
    }


def evaluate(
    test_jsonl: Path,
    output_dir: Path,
    adapter_path: Optional[Path],
    label: str,
    device: str,
    limit: Optional[int] = None,
) -> dict:
    print(f"Loading model (label={label}, adapter={adapter_path}, device={device})...")
    model, tokenizer = load_model(adapter_path, device)

    print(f"Loading test set from {test_jsonl}")
    ds = load_dataset("json", data_files=str(test_jsonl), split="train")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    n = len(ds)
    predictions: list[str] = []
    references: list[str] = []
    start = time.time()
    print(f"Generating {n} commit messages...")
    for i, row in enumerate(ds):
        pred = generate_one(model, tokenizer, row["diff"])
        predictions.append(pred)
        references.append(row["message"])
        if (i + 1) % 25 == 0 or i + 1 == n:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{n}] {rate:.2f} samples/s  ETA {eta/60:.1f} min  "
                f"last: {pred[:80]!r}"
            )

    metrics = compute_metrics(predictions, references)
    metrics["wall_clock_sec"] = round(time.time() - start, 1)
    print("Metrics:", json.dumps(metrics, indent=2))

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"metrics_{label}.json"
    preds_path = output_dir / f"predictions_{label}.jsonl"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    with preds_path.open("w", encoding="utf-8") as f:
        for diff_row, pred in zip(ds, predictions):
            record = {
                "diff": diff_row["diff"],
                "reference": diff_row["message"],
                "prediction": pred,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {metrics_path} and {preds_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--label", type=str, required=True, help="e.g. 'base' or 'tuned'"
    )
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap test examples for a quick smoke run",
    )
    args = parser.parse_args()

    evaluate(
        test_jsonl=args.test_jsonl,
        output_dir=args.output_dir,
        adapter_path=args.adapter,
        label=args.label,
        device=args.device,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

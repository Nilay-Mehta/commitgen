# commitgen

A small language model fine-tuned with LoRA to generate conventional-commit-style messages from git diffs. Base model: **Qwen2.5-Coder-0.5B-Instruct**. Trained on Google Colab's free T4. Deployed locally via Ollama. Integrates with [reviewbot](https://github.com/Nilay-Mehta/reviewbot) as a `commit-msg` subcommand.

## Status

Training complete (Day 3). Evaluation pending (Day 4). Deployment + integration pending (Days 5–6). Build plan and decisions live in `../project_2_finetune_commit_msg.md`.

## Dataset

Source: [Maxscha/commitbench](https://huggingface.co/datasets/Maxscha/commitbench) — a large public corpus of `(diff, commit_message)` pairs scraped from GitHub.

Filtered down to clean, learnable examples:

| Filter | Rule |
|---|---|
| Language | `diff_languages` ⊂ `{py, js, ts, go, java}` (every file in the diff must qualify) |
| Message format | First line matches `^(feat\|fix\|docs\|style\|refactor\|test\|chore)(\(scope\))?!?: .+` |
| Message length | 5–200 chars |
| Diff length | ≤ 5000 chars and ≤ 1500 Qwen2.5-Coder tokens |
| Anonymized literals | Reject any message containing `<I>` (CommitBench's placeholder for numbers) |

Splits: **15 000 train / 500 eval / 500 test** (shuffled with `seed=42`).

See `data/prepare_dataset.py` for the full pipeline.

## Training

**Method:** LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`. Base model loaded in `bfloat16` (not 4-bit) — at 0.5B params, QLoRA saved little VRAM on a T4 and dragged in the bitsandbytes/Triton/CUDA version chain.

**Hyperparameters:**

| | |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Trainable params | 2.16M (0.44% of 496M) |
| Optimizer | AdamW (torch) |
| Learning rate | 2e-4, cosine schedule, 3% warmup |
| Effective batch size | 16 (`per_device=4` × `grad_accum=4`) |
| Max sequence length | 2048 |
| Precision | bf16, gradient checkpointing on |
| Epochs (planned) | 3 |
| Seed | 42 |

**Hardware:** single Google Colab free-tier T4 (15 GB VRAM). Wall-clock ≈ 6 hours for a full run. bf16 on T4 doesn't hit tensor cores (Turing arch), so throughput was ~8 sec/step.

**Logging:** Weights & Biases, [project: commitgen](https://wandb.ai/nilaymehta1405-poornima-college-of-engineering/commitgen).

## Results

_Filled in at end of Day 4._

| Model | BLEU-4 | ROUGE-L | Type Acc | Avg Length |
|---|---|---|---|---|
| Qwen2.5-Coder-0.5B base (zero-shot) | — | — | — | — |
| Qwen2.5-Coder-0.5B + LoRA (ours)    | — | — | — | — |

## Layout

```
commitgen/
├── data/            # prepare_dataset.py + train/eval/test JSONL (gitignored)
├── training/        # Colab notebooks + standalone train script
├── evaluation/      # BLEU/ROUGE/type-accuracy + human eval
└── deployment/      # LoRA merge → GGUF → Ollama Modelfile
```

## Reproduce

_Filled in at end of Day 7._

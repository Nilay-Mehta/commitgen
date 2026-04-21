# commitgen

A small language model fine-tuned with LoRA to generate conventional-commit-style messages from git diffs. Base model: Qwen2.5-Coder-0.5B-Instruct. Trained on Google Colab's free T4. Deployed locally via Ollama. Integrates with [reviewbot](https://github.com/Nilay-Mehta/reviewbot) as a `commit-msg` subcommand.

## Status

In progress — build plan and decisions live in `../project_2_finetune_commit_msg.md`.

## Results

_Eval table goes here once Day 4 is done._

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

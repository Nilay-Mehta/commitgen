# commitgen

`commitgen` is a local-first commit message generator. It fine-tunes
Qwen2.5-Coder-0.5B-Instruct with LoRA to turn Git diffs into concise
conventional commit messages, then deploys the tuned model locally through
Ollama and a small CLI.

The repo also includes a hosted static web wrapper for portfolio review. The
website is a deterministic browser demo; the real model workflow runs locally
through the CLI.

Repository: https://github.com/Nilay-Mehta/commitgen

Live demo: https://commitgen-ten.vercel.app/

## Highlights

- Fine-tuned Qwen2.5-Coder-0.5B-Instruct with LoRA on filtered CommitBench data.
- Built a deterministic train/eval/test dataset pipeline from raw GitHub diffs.
- Evaluated against the base model with BLEU-4, ROUGE-L, type accuracy, and output length.
- Merged the LoRA adapter, converted to GGUF, quantized to Q4_K_M, and served with Ollama.
- Shipped a local CLI: `commitgen msg`, `commitgen commit`, and `commitgen push`.
- Added a static Vercel-ready web wrapper for interviews and portfolio review.

## Demo And Local Install

The web wrapper lives in [web/index.html](web/index.html). It is designed to be
deployed on Vercel from this repo. The root route is handled by
[vercel.json](vercel.json).

For the full local model workflow, follow [INSTALL.md](INSTALL.md).

Quick local CLI flow:

```powershell
git clone https://github.com/Nilay-Mehta/commitgen.git
cd commitgen

# Add deployment/commitgen-Q4_K_M.gguf before this step.
ollama create commitgen -f deployment/Modelfile
pip install -e deployment

git add .
commitgen commit
```

## Dataset

Source: [Maxscha/commitbench](https://huggingface.co/datasets/Maxscha/commitbench),
a public corpus of `(diff, commit_message)` pairs scraped from GitHub.

The dataset pipeline filters for clean, learnable examples:

| Filter | Rule |
|---|---|
| Language | Every `diff_languages` token must be in `{py, js, ts, go, java}` |
| Message format | First line must be a conventional commit type: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, or `chore` |
| Message length | 5-200 characters |
| Diff length | Up to 5000 chars and up to 1500 Qwen2.5-Coder tokens |
| Anonymized literals | Reject messages containing `<I>` |

Splits:

| Split | Rows |
|---|---:|
| Train | 15,000 |
| Eval | 500 |
| Test | 500 |

Implementation: [data/prepare_dataset.py](data/prepare_dataset.py)

## Training

Training uses LoRA on the attention projection layers:

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Trainable params | 2.16M, about 0.44% of 496M |
| Optimizer | AdamW torch |
| Learning rate | 2e-4 |
| Scheduler | Cosine, 3% warmup |
| Effective batch size | 16 |
| Max sequence length | 2048 |
| Precision | bf16 |
| Seed | 42 |

The final training code loads the 0.5B model in bf16 rather than 4-bit. On a
free Colab T4, this avoided bitsandbytes/Triton friction while still fitting
comfortably in memory.

Implementation: [training/train_lora.py](training/train_lora.py)

## Results

Evaluation used the untouched 500-example test split with greedy decoding and
`max_new_tokens=64`. The base model is Qwen2.5-Coder-0.5B-Instruct with the same
prompt template and no adapter.

| Model | BLEU-4 | ROUGE-L | Type Acc | Avg Length |
|---|---:|---:|---:|---:|
| Base zero-shot | 1.92 | 15.88 | 0.80% | 61.5 chars |
| LoRA tuned | 10.77 | 28.93 | 58.20% | 58.3 chars |
| Delta | +8.85 | +13.05 | +57.4 pp | closer to 54.8-char refs |

Type accuracy means the generated conventional-commit prefix matches the
reference prefix. The base model rarely follows the required format; the tuned
adapter learns it reliably.

Implementation: [evaluation/evaluate.py](evaluation/evaluate.py)

## Deployment

The trained LoRA adapter is merged into the base model, converted to GGUF,
quantized to Q4_K_M, and loaded by Ollama through [deployment/Modelfile](deployment/Modelfile).

Deployment pieces:

- [deployment/merge_lora.py](deployment/merge_lora.py): merge LoRA adapter into the base model.
- [deployment/deploy.ipynb](deployment/deploy.ipynb): Colab workflow for GGUF conversion and quantization.
- [deployment/Modelfile](deployment/Modelfile): Ollama runtime config and ChatML template.
- [deployment/commitgen_cli.py](deployment/commitgen_cli.py): CLI wrapper around the local Ollama model.

The quantized model artifact is intentionally not tracked in Git:

```text
deployment/commitgen-Q4_K_M.gguf
```

## CLI

Install locally:

```powershell
pip install -e deployment
```

Commands:

```powershell
commitgen msg     # print a message for staged changes
commitgen commit  # generate, confirm/edit, then commit
commitgen push    # commit, then git push
```

The CLI always uses staged Git diffs and asks before committing.

## Project Layout

```text
commitgen/
  data/          Dataset preparation
  training/      Colab notebooks and LoRA training code
  evaluation/    Metrics and prediction generation
  deployment/    LoRA merge, GGUF/Ollama deployment, CLI
  web/           Static portfolio wrapper
  INSTALL.md     Local setup guide
  vercel.json    Static web deployment routing
```

## Reproduce

1. Build the dataset:

```powershell
python data/prepare_dataset.py
```

2. Train in Colab:

```text
training/train_lora.ipynb
```

3. Evaluate:

```text
evaluation/evaluate.ipynb
```

4. Merge and deploy:

```text
deployment/deploy.ipynb
```

5. Install and use the CLI:

```powershell
pip install -e deployment
commitgen commit
```

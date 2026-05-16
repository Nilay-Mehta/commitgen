# commitgen Installation Guide

The hosted website is a portfolio demo. The local installation runs the real
fine-tuned model through Ollama and installs the `commitgen` CLI for Git
workflows.

Repository: https://github.com/Nilay-Mehta/commitgen

## Requirements

- Python 3.10+
- Git
- Ollama
- The quantized model file: `commitgen-Q4_K_M.gguf`

## 1. Clone the Repository

```powershell
git clone https://github.com/Nilay-Mehta/commitgen.git
cd commitgen
```

## 2. Add the Model Artifact

Place `commitgen-Q4_K_M.gguf` in the `deployment/` folder, next to
`deployment/Modelfile`.

Expected layout:

```text
commitgen/
  deployment/
    Modelfile
    commitgen-Q4_K_M.gguf
```

## 3. Register the Ollama Model

```powershell
ollama create commitgen -f deployment/Modelfile
```

Quick check:

```powershell
ollama run commitgen "Write a conventional-commit message for this diff:"
```

## 4. Install the CLI

From the repo root:

```powershell
pip install -e deployment
```

Verify:

```powershell
commitgen --help
```

## 5. Use commitgen in a Git Repo

Preview a message from staged changes:

```powershell
git add .
commitgen msg
```

Generate, confirm, optionally edit, and commit:

```powershell
commitgen commit
```

Generate, commit, and push:

```powershell
commitgen push
```

## Optional Web Preview

From the repo root:

```powershell
python -m http.server 8000
```

Open:

```text
http://localhost:8000/web/
```

If Ollama is running locally with the `commitgen` model, the page can call the
local model. Otherwise it uses the built-in browser preview.

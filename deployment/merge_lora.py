"""Merge the trained LoRA adapter into the base Qwen2.5-Coder-0.5B and save as
a standalone HuggingFace model directory, ready for GGUF conversion.

The merge produces a single fp16 model where the LoRA weights are folded
into the base weights, so downstream tooling (llama.cpp's convert script,
Ollama) doesn't need to know LoRA exists.

CLI:
    python deployment/merge_lora.py \\
        --adapter /content/drive/MyDrive/commitgen_checkpoints/qwen25coder-lora-full-0/checkpoint-2000 \\
        --output /content/commitgen/deployment/merged_model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Path to the LoRA adapter directory (containing adapter_config.json + adapter_model.safetensors)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the merged model (a directory)",
    )
    args = parser.parse_args()

    print(f"Loading base model {BASE_MODEL}...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    print(f"Loading adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(base, str(args.adapter))

    print("Merging adapter into base weights (merge_and_unload)...")
    merged = model.merge_and_unload()

    print("Loading tokenizer (from base)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {args.output}...")
    merged.save_pretrained(str(args.output), safe_serialization=True)
    tokenizer.save_pretrained(str(args.output))
    print(f"Done. Files written to {args.output}:")
    for p in sorted(args.output.iterdir()):
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

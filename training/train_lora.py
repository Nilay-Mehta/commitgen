from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
import wandb
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import trl

print(f"trl version: {trl.__version__}")

try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None

BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
MAX_SEQ_LEN = 2048
SYSTEM_PROMPT = (
    "You are a helpful assistant that writes concise conventional-commit "
    "messages from git diffs. Output only the commit message, no extra "
    "commentary."
)
USER_TEMPLATE = "Write a conventional-commit message for this diff:\n\n{diff}"


def load_splits(data_dir: Path) -> DatasetDict:
    train_path = data_dir / "train.jsonl"
    eval_path = data_dir / "eval.jsonl"
    train_ds = load_dataset("json", data_files=str(train_path), split="train")
    eval_ds = load_dataset("json", data_files=str(eval_path), split="train")
    return DatasetDict({"train": train_ds, "eval": eval_ds})


def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def make_formatting_func(tokenizer):
    def _format(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(diff=example["diff"])},
            {"role": "assistant", "content": example["message"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    return _format


def _next_version_path(path: Path) -> Path:
    if not path.exists():
        return path
    version = 2
    while True:
        candidate = path.parent / f"{path.name}-v{version}"
        if not candidate.exists():
            return candidate
        version += 1


def _build_training_args(
    output_dir: Path, mode: Literal["smoke", "full"], run_name: str
):
    is_smoke = mode == "smoke"
    common_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=3 if not is_smoke else 1,
        max_steps=200 if is_smoke else -1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=25,
        save_strategy="steps",
        save_steps=100 if is_smoke else 500,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=100 if is_smoke else 500,
        report_to="wandb",
        run_name=run_name,
        seed=42,
    )
    return TrainingArguments(**common_kwargs)


def run_training(
    mode: Literal["smoke", "full"], data_dir: Path, drive_root: Path, run_name: str
):
    if mode not in {"smoke", "full"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY is not set. Run `wandb login` in Colab before training."
        )

    output_dir = drive_root / "commitgen_checkpoints" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = _next_version_path(output_dir / "final")

    is_smoke = mode == "smoke"
    wandb.init(
        project="commitgen",
        name=run_name,
        config={
            "mode": mode,
            "base_model": BASE_MODEL,
            "max_seq_len": MAX_SEQ_LEN,
            "num_train_epochs": 3 if not is_smoke else 1,
            "max_steps": 200 if is_smoke else -1,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "save_steps": 100 if is_smoke else 500,
            "eval_steps": 100 if is_smoke else 500,
        },
    )

    try:
        splits = load_splits(data_dir)
        model, tokenizer = load_model_and_tokenizer()
        response_template = "<|im_start|>assistant\n"
        collator = DataCollatorForCompletionOnlyLM(
            response_template, tokenizer=tokenizer
        )

        args = _build_training_args(output_dir, mode, run_name)

        try:
            trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=splits["train"],
                eval_dataset=splits["eval"],
                tokenizer=tokenizer,
                formatting_func=make_formatting_func(tokenizer),
                data_collator=collator,
                max_seq_length=MAX_SEQ_LEN,
            )
        except TypeError:
            if SFTConfig is None:
                raise
            sft_args = SFTConfig(
                output_dir=str(output_dir),
                num_train_epochs=3 if not is_smoke else 1,
                max_steps=200 if is_smoke else -1,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                lr_scheduler_type="cosine",
                warmup_ratio=0.03,
                bf16=True,
                optim="paged_adamw_8bit",
                logging_steps=25,
                save_strategy="steps",
                save_steps=100 if is_smoke else 500,
                save_total_limit=2,
                eval_strategy="steps",
                eval_steps=100 if is_smoke else 500,
                report_to="wandb",
                run_name=run_name,
                seed=42,
                max_seq_length=MAX_SEQ_LEN,
            )
            trainer = SFTTrainer(
                model=model,
                args=sft_args,
                train_dataset=splits["train"],
                eval_dataset=splits["eval"],
                tokenizer=tokenizer,
                formatting_func=make_formatting_func(tokenizer),
                data_collator=collator,
            )

        trainer.train()
        trainer.save_model(str(final_dir))
    finally:
        wandb.finish()

    return final_dir

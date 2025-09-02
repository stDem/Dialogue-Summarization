#!/usr/bin/env python
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType

def parse_args():
    p = argparse.ArgumentParser(
        description="LoRA fine-tuning on DialogSum (FLAN-T5 + PEFT)"
    )
    p.add_argument("--base_model", type=str, default="google/flan-t5-small")
    p.add_argument("--output_dir", type=str, default="artifacts/flan_t5_small_lora_dialogsum")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_source_len", type=int, default=1024)
    p.add_argument("--max_target_len", type=int, default=150)
    # (train_file is not needed; we load HF directly)
    return p.parse_args()

def main():
    args = parse_args()

    # Device pick (CUDA > MPS > CPU)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # Tokenizer & base model first (so tokenizer is in scope for preprocess)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    # LoRA config
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q","k","v","o"],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(base_model, lora_cfg)

    # Move to device
    model = model.to(device)

    # Load DialogSum
    raw = load_dataset("knkarthick/dialogsum")
    train_ds = raw["train"]
    val_ds   = raw["validation"] if "validation" in raw else raw["test"]

    PREFIX = (
        "Write a concise third-person summary of the conversation. "
        "Be brief, preserve key facts and names, and avoid first-person narration.\n\n"
    )

    def preprocess(examples):
        # examples: dict with keys including 'dialogue' and 'summary'
        inputs = [PREFIX + d for d in examples["dialogue"]]
        model_inputs = tokenizer(inputs, max_length=args.max_source_len, truncation=True)
        # target/labels
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["summary"],
                max_length=args.max_target_len,
                truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    tokenized_val   = val_ds.map(preprocess,   batched=True, remove_columns=val_ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available(),
        fp16=False if torch.backends.mps.is_available() else (not torch.cuda.is_available()),
        optim="adamw_torch",
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Save only LoRA adapter + tokenizer (adapter merges into base at inference via PEFT)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")

if __name__ == "__main__":
    main()

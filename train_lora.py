#!/usr/bin/env python
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import torch

def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tuning for dialogue summarization (FLAN-T5 + PEFT)")
    p.add_argument("--base_model", type=str, default="google/flan-t5-small")
    p.add_argument("--train_file", type=str, required=True, help="JSONL with fields: dialog, summary")
    p.add_argument("--output_dir", type=str, default="artifacts/flan_t5_small_lora")
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_source_len", type=int, default=1024)
    p.add_argument("--max_target_len", type=int, default=200)
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    model = model.to(device)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q","k","v","o"],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, lora_cfg)

    ds = load_dataset("json", data_files={"train": args.train_file})

    prefix = "Summarize the following dialogue briefly:\n\n"
    def preprocess(examples):
        inputs = [prefix + d for d in examples["dialog"]]
        model_inputs = tokenizer(inputs, max_length=args.max_source_len, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], max_length=args.max_target_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        gradient_accumulation_steps=1,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
        optim="adamw_torch",
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")

if __name__ == "__main__":
    main()

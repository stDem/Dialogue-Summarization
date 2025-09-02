#!/usr/bin/env python
import argparse, random
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType

def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tuning on DialogSum (car-filtered, GPU/MPS/CPU-aware)")
    p.add_argument("--base_model", type=str, default="google/flan-t5-small")
    p.add_argument("--output_dir", type=str, default="artifacts/flan_t5_small_lora_dialogsum_car")
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_source_len", type=int, default=768)
    p.add_argument("--max_target_len", type=int, default=120)
    p.add_argument("--topics", type=str, default="car,auto,vehicle,driver,parking,traffic")
    p.add_argument("--match_in_dialog", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=1200)
    p.add_argument("--max_val_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    has_cuda = torch.cuda.is_available()
    has_mps  = torch.backends.mps.is_available()
    device = "cuda" if has_cuda else ("mps" if has_mps else "cpu")
    print(f"[INFO] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q","k","v","o"],
        bias="none", task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(base_model, lora_cfg).to(device)

    raw = load_dataset("knkarthick/dialogsum")
    train_ds = raw["train"]
    val_ds   = raw["validation"] if "validation" in raw else raw["test"]

    keywords = [k.strip().lower() for k in args.topics.split(",") if k.strip()]
    def keep_row(ex):
        if not keywords: return True
        topic = (ex.get("topic") or "").lower()
        hit_topic = any(k in topic for k in keywords)
        if args.match_in_dialog:
            dialog = (ex.get("dialogue") or "").lower()
            hit_dialog = any(k in dialog for k in keywords)
            return hit_topic and hit_dialog
        return hit_topic

    if keywords:
        print(f"[INFO] Filtering with keywords={keywords}, match_in_dialog={args.match_in_dialog}")
        train_ds = train_ds.filter(keep_row)
        val_ds   = val_ds.filter(keep_row)
        print(f"[INFO] After filter -> train: {len(train_ds)}, val: {len(val_ds)}")

    def subsample(ds, n):
        if n is None or n < 0 or n >= len(ds): return ds
        idx = list(range(len(ds))); random.shuffle(idx)
        return ds.select(idx[:n])

    train_ds = subsample(train_ds, args.max_train_samples)
    val_ds   = subsample(val_ds, args.max_val_samples)
    print(f"[INFO] Using train={len(train_ds)} examples, val={len(val_ds)} examples")

    PREFIX = ("Write a concise third-person summary of the conversation. "
              "Be brief, preserve key facts and names, and avoid first-person narration.\\n\\n")

    def preprocess(examples):
        inputs = [PREFIX + d for d in examples["dialogue"]]
        model_inputs = tokenizer(inputs, max_length=args.max_source_len, truncation=True)
        labels = tokenizer(text_target=examples["summary"], max_length=args.max_target_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names, num_proc=1)
    tokenized_val   = val_ds.map(preprocess,   batched=True, remove_columns=val_ds.column_names,   num_proc=1)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def _move_to_device(obj, target):
        if torch.is_tensor(obj): return obj.to(target)
        if isinstance(obj, dict): return {k: _move_to_device(v, target) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            seq = [_move_to_device(x, target) for x in obj]
            return type(obj)(seq) if isinstance(obj, tuple) else seq
        return obj

    class SafeTrainer(Trainer):
        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            target = next(self.model.parameters()).device
            return _move_to_device(inputs, target)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
        fp16=False if has_mps else (not has_cuda),
        bf16=True if (has_cuda and torch.cuda.is_bf16_supported()) else False,
        optim="adamw_torch",
        report_to=[]
    )

    trainer = SafeTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[INFO] Saved LoRA adapter to {args.output_dir}")

if __name__ == "__main__":
    main()

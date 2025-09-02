#!/usr/bin/env python
import argparse, random, math, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tuning — manual loop (car-filtered, GPU/MPS/CPU)")
    p.add_argument("--base_model", type=str, default="google/flan-t5-small")
    p.add_argument("--output_dir", type=str, default="artifacts/flan_t5_small_lora_dialogsum_car")
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_source_len", type=int, default=768)
    p.add_argument("--max_target_len", type=int, default=120)
    p.add_argument("--device", type=str, default="auto", choices=["auto","cuda","mps","cpu"])
    p.add_argument("--topics", type=str, default="car,auto,vehicle,driver,parking,traffic")
    p.add_argument("--match_in_dialog", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=1200)
    p.add_argument("--max_val_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_accum", type=int, default=1)
    return p.parse_args()

def pick_device(pref):
    if pref != "auto":
        return pref
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"[INFO] Using device: {device}")

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q","k","v","o"],
        bias="none", task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(base, lora_cfg).to(device)

    # Load DialogSum
    raw = load_dataset("knkarthick/dialogsum")
    train = raw["train"]
    val   = raw["validation"] if "validation" in raw else raw["test"]

    # Filter
    keys = [k.strip().lower() for k in args.topics.split(",") if k.strip()]
    def keep(ex):
        if not keys: return True
        topic = (ex.get("topic") or "").lower()
        hit_topic = any(k in topic for k in keys)
        if args.match_in_dialog:
            dlg = (ex.get("dialogue") or "").lower()
            hit_dlg = any(k in dlg for k in keys)
            return hit_topic and hit_dlg
        return hit_topic
    if keys:
        print(f"[INFO] Filtering with keywords={keys}, match_in_dialog={args.match_in_dialog}")
        train = train.filter(keep); val = val.filter(keep)
        print(f"[INFO] After filter -> train: {len(train)}, val: {len(val)}")

    # Subsample
    def select(ds, n):
        if n is None or n < 0 or n >= len(ds): return ds
        idx = list(range(len(ds))); random.shuffle(idx)
        return ds.select(idx[:n])
    train = select(train, args.max_train_samples)
    val   = select(val, args.max_val_samples)
    print(f"[INFO] Using train={len(train)}, val={len(val)}")

    PREFIX = ("Write a concise third-person summary of the conversation. "
              "Be brief, preserve key facts and names, and avoid first-person narration.\n\n")

    def encode_batch(batch):
        inputs = [PREFIX + d for d in batch["dialogue"]]
        model_inputs = tok(inputs, max_length=args.max_source_len, truncation=True, padding=True)
        with tok.as_target_tokenizer():
            labels = tok(batch["summary"], max_length=args.max_target_len, truncation=True, padding=True)
        model_inputs["labels"] = labels["input_ids"]
        # attention mask for labels is not required for T5 loss; set -100 for pad tokens
        pad = tok.pad_token_id
        model_inputs["labels"] = [[(lid if lid != pad else -100) for lid in seq] for seq in model_inputs["labels"]]
        return model_inputs

    train_enc = train.map(encode_batch, batched=True, remove_columns=train.column_names)
    val_enc   = val.map(encode_batch,   batched=True, remove_columns=val.column_names)

    # Torch dataset wrappers
    class TDS(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            ex = self.ds[i]
            item = {k: torch.tensor(ex[k]) for k in ["input_ids","attention_mask","labels"]}
            return item

    train_dl = DataLoader(TDS(train_enc), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_dl   = DataLoader(TDS(val_enc),   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_dl) / args.grad_accum) * args.num_epochs
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(total_steps*args.warmup_ratio), num_training_steps=total_steps)

    def step_batch(batch):
        for k in batch:
            batch[k] = batch[k].to(device)
        out = model(**batch)
        loss = out.loss / args.grad_accum
        loss.backward()
        return loss.item()

    def eval_loop():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                for k in batch: batch[k] = batch[k].to(device)
                out = model(**batch)
                tot += out.loss.item(); n += 1
        model.train()
        return tot / max(n,1)

    model.train()
    global_step = 0
    for epoch in range(1, args.num_epochs+1):
        running = 0.0
        for i, batch in enumerate(train_dl, 1):
            loss = step_batch(batch)
            running += loss
            if i % args.grad_accum == 0:
                if device == "mps":
                    torch.mps.synchronize()
                optim.step(); sched.step(); optim.zero_grad()
                global_step += 1
            if global_step % 20 == 0:
                print(f"[epoch {epoch}] step {global_step} | loss={running / max(1,(i)):.4f}")
        val_loss = eval_loop()
        print(f"[epoch {epoch}] val_loss={val_loss:.4f}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"[INFO] Saved LoRA adapter to {args.output_dir}")

if __name__ == "__main__":
    main()

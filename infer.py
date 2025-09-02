#!/usr/bin/env python
import argparse, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

def build_model(base_model:str, lora_dir:str|None):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    if lora_dir:
        model = PeftModel.from_pretrained(model, lora_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

def summarize_dialog(tokenizer, model, device, dialog:str, max_new_tokens:int=120):
    prompt = "Summarize the following dialogue briefly:\n\n" + dialog.strip()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="google/flan-t5-small")
    p.add_argument("--lora_dir", default=None)
    p.add_argument("--dialog_file", required=True)
    args = p.parse_args()
    tok, mdl, dev = build_model(args.base_model, args.lora_dir)
    dialog = open(args.dialog_file, "r", encoding="utf-8").read()
    print(summarize_dialog(tok, mdl, dev, dialog))

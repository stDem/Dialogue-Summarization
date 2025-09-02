#!/usr/bin/env python
import argparse, os, re, torch, datetime, traceback
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

app = Flask(__name__, static_folder="static", static_url_path="/static")

tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model_ready = False

def log_ex(e: Exception):
    app.logger.error("Exception: %s\n%s", e, traceback.format_exc())

def load_models(base_model: str, lora_dir: str | None):
    global tokenizer, model, model_ready
    try:
        app.logger.info(f"Loading base model: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        if lora_dir and os.path.isdir(lora_dir):
            app.logger.info(f"Loading LoRA adapter from: {lora_dir}")
            base = PeftModel.from_pretrained(base, lora_dir)
        base = base.to(device)
        base.eval()
        model = base
        model_ready = True
        app.logger.info(f"Model ready on device: {device}")
    except Exception as e:
        log_ex(e)
        model_ready = False

def build_chat_prompt(turns):
    lines = []
    for t in (turns[-8:] if len(turns) > 8 else turns):
        role = t.get("role","user")
        text = (t.get("text") or "").strip().replace("\n", " ")
        lines.append(f"#Assistant#: {text}" if role=="assistant" else f"#User#: {text}")
    return (
        "Continue this casual conversation. Respond as #Assistant# with one short, helpful sentence.\n\n"
        + "\n".join(lines) + "\n#Assistant#:"
    )

def generate_reply(turns, max_new_tokens=64):
    if not model_ready:
        return "Model not ready yet. Try again in a moment."
    prompt = build_chat_prompt(turns)
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if device in ("cuda","mps"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True, top_p=0.9, temperature=0.9
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        reply = text.split("#Assistant#:")[-1].strip()
        return (reply or "Okay!").split("\n")[0].strip()
    except Exception as e:
        log_ex(e)
        return "Error generating reply."

def to_dialog_text(turns):
    lines = []
    for t in turns:
        role = t.get("role","user")
        spk = "Person1" if role == "user" else "Person2"
        txt = (t.get("text") or "").replace("\n"," ").strip()
        lines.append(f"#{spk}#: {txt}")
    return "\n".join(lines)

def summarize_dialog_text(dialog_text:str, max_new_tokens:int=150)->str:
    if not model_ready:
        return "(Model not ready) Example summary: Two people discuss a topic and agree next steps."
    prompt = (
        "Write a concise third-person summary of the conversation. "
        "Be brief, preserve key facts and names, and avoid first-person narration.\n\n"
        + dialog_text.strip()
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if device in ("cuda","mps"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        log_ex(e)
        return "Error creating summary."

import re as _re
def parse_dialog_file(text:str):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        m = _re.match(r"#([^#]+)#:\s*(.*)$", line)
        if m:
            speaker = m.group(1).strip()
            msg = m.group(2).strip()
            role = "user" if speaker.lower() in ["person1","user","alice"] else "assistant"
            rows.append({"role": role, "speaker": speaker, "text": msg, "ts": datetime.datetime.now().isoformat()})
        else:
            rows.append({"role": "assistant", "speaker": "Narrator", "text": line, "ts": datetime.datetime.now().isoformat()})
    return rows

from flask import send_from_directory

@app.get("/")
def index():
    return app.send_static_file("index.html")

@app.get("/api/health")
def health():
    return jsonify({"ok": True, "model_ready": model_ready, "device": device})

@app.post("/api/chat")
def api_chat():
    try:
        data = request.get_json(force=True) or {}
        turns = data.get("turns", [])
        reply = generate_reply(turns)
        return jsonify({"reply": reply})
    except Exception as e:
        log_ex(e)
        return jsonify({"error":"chat_failed","detail":str(e)}), 500

@app.post("/api/summarize")
def api_summarize():
    try:
        data = request.get_json(force=True) or {}
        turns = data.get("turns", [])
        dialog = to_dialog_text(turns)
        summary = summarize_dialog_text(dialog)
        return jsonify({"summary": summary})
    except Exception as e:
        log_ex(e)
        return jsonify({"error":"summarize_failed","detail":str(e)}), 500

@app.post("/api/load_dialog")
def api_load_dialog():
    try:
        if "file" not in request.files:
            return jsonify({"error":"no_file"}), 400
        f = request.files["file"]
        text = f.read().decode("utf-8", errors="ignore")
        parsed = parse_dialog_file(text)
        return jsonify({"turns": parsed})
    except Exception as e:
        log_ex(e)
        return jsonify({"error":"load_failed","detail":str(e)}), 500

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base_model", default="google/flan-t5-small")
    parser.add_argument("--lora_dir", default=None)
    args = parser.parse_args()
    load_models(args.base_model, args.lora_dir)
    app.run(host=args.host, port=args.port, debug=True)

if __name__ == "__main__":
    main()

# FLAN-T5 + PEFT LoRA — Dialogue Summarizer (Full v2)

End-to-end prototype that fine-tunes **FLAN-T5** with **LoRA (PEFT)** for dialogue summarization and serves a local web UI (vanilla HTML/JS) via **Flask**.

## Contents
- `train_lora.py` — LoRA fine-tuning for seq2seq on (dialog, summary).
- `infer.py` — inference helper that loads the base model and an optional LoRA adapter.
- `app.py` — Flask backend with endpoints for chat, summarize, and loading a dialog from a text file.
- `static/index.html`, `static/script.js`, `static/styles.css` — simple two-column chat UI + persistent summary.
- `data/sample_train.jsonl` — tiny dataset to verify training loop.
- `sample_dialog.txt` — example dialog file for upload.
- `artifacts/` — folder where LoRA adapters are saved.

## Quickstart

> Use Python 3.10+ in a fresh venv.

```bash
pip install -r requirements.txt
```

### 1) (Optional) Train a LoRA adapter
By default we use `google/flan-t5-small` for speed. You can switch to `flan-t5-base`.

```bash
python train_lora.py   --base_model google/flan-t5-small   --output_dir artifacts/flan_t5_small_lora   --train_file data/sample_train.jsonl   --num_epochs 1   --batch_size 4   --lr 2e-4
```

### 2) Run the app
```bash
python app.py --host 127.0.0.1 --port 8000   --base_model google/flan-t5-small   --lora_dir artifacts/flan_t5_small_lora
```

Open http://127.0.0.1:8000

### UI
- Normal chat with a local FLAN-T5 small-talk reply (free/offline).
- **Load dialog (.txt)**: renders transcript as bubbles you can continue from.
- **Summarize**: generates/upgrades the summary of the current chat.

### Data format
- Training file (`jsonl`): each line has `"dialog"` and `"summary"` fields.
- Dialog text file (`.txt`) for loading: lines like `#Person1#: Hello`, `#Person2#: Hi!`


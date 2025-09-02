# Dialogue Summarizer — FLAN‑T5 + LoRA (car topics) — v4

Same UI you liked. Two training options:
1) `train_lora_manual.py` — **recommended on Mac (MPS GPU)**. Manual loop that reliably keeps tensors on the same device.
2) `train_lora.py` — Trainer-based (kept for reference).

## Quickstart
pip install -r requirements.txt

### Manual training (recommended on Mac GPU/MPS)
python train_lora_manual.py   --base_model google/flan-t5-base   --output_dir artifacts/flan_t5_base_lora_dialogsum_car   --topics "car,auto,vehicle,driver,parking,traffic"   --max_train_samples 1200   --max_val_samples 200   --num_epochs 2   --batch_size 8

(Force CPU if needed: add `--device cpu`)

### Serve the app
python app.py --host 127.0.0.1 --port 8000   --base_model google/flan-t5-base   --lora_dir artifacts/flan_t5_base_lora_dialogsum_car

# Dialogue Summarizer — FLAN‑T5 + LoRA (car topics)

## Quickstart
pip install -r requirements.txt

### Manual training
python train_lora_manual.py   --base_model google/flan-t5-base   --output_dir artifacts/flan_t5_base_lora_dialogsum_car   --topics "car,auto,vehicle,driver,parking,traffic"   --max_train_samples 1200   --max_val_samples 200   --num_epochs 2   --batch_size 8

### Serve the app
python app.py --host 127.0.0.1 --port 8000   --base_model google/flan-t5-base   --lora_dir artifacts/flan_t5_base_lora_dialogsum_car

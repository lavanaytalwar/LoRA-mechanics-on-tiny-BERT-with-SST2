# LoRA reproduction 

This repo reproduces the core mechanics and practical findings from **"LoRA: Low-Rank Adaptation of Large Language Models"** in a lightweight setting:

- **Custom LoRA implementation** (rank decomposition, frozen base weights, trainable adapters, merge/unmerge)
- **Full fine-tuning vs LoRA fine-tuning** comparison
- **Rank sweep**: r = 2, 4, 8, 16
- **Metrics + plots + report** generated automatically

Target task: **SST-2** sentiment classification using a tiny BERT model (CPU-friendly).

## Setup

From the project root:

```bash
cd lora_reproduction
# Use Python 3.11 (PyTorch-compatible on Windows)
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run experiments (full FT + LoRA rank sweep)

```bash
py -3.11 -m src.train --config configs\sst2_bert_tiny.yaml --mode full
py -3.11 -m src.train --config configs\sst2_bert_tiny.yaml --mode lora --lora_r 2
py -3.11 -m src.train --config configs\sst2_bert_tiny.yaml --mode lora --lora_r 4
py -3.11 -m src.train --config configs\sst2_bert_tiny.yaml --mode lora --lora_r 8
py -3.11 -m src.train --config configs\sst2_bert_tiny.yaml --mode lora --lora_r 16

py -3.11 -m src.evaluate --config configs\sst2_bert_tiny.yaml --runs_dir outputs\runs
py -3.11 -m scripts.make_report --runs_dir outputs\runs
```

Or run the full pipeline in one go (from inside the venv):

```bash
.\.venv\Scripts\python -m scripts.run_experiments --config configs\sst2_bert_tiny.yaml --ranks 2,4,8,16 --with_peft
```

Outputs:
- `outputs/runs/<run_name>/metrics.json`
- `outputs/metrics.csv`
- `outputs/plots/*.png`
- `results.md`

## Inference (single example)

```bash
py -3.11 -m src.inference --run_dir outputs\runs\<run_name> --text "this movie was surprisingly good"
```

## Notes

- Designed for **Windows + CPU-only**.
- Uses small dataset subsets by default for speed; increase in `configs/*.yaml` if you have more time/RAM.


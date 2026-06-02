# LoRA reproduction 

This repo reproduces the core mechanics and practical findings from **"LoRA: Low-Rank Adaptation of Large Language Models"** in a lightweight setting:

- **Custom LoRA implementation** (rank decomposition, frozen base weights, trainable adapters, merge/unmerge)
- **Full fine-tuning vs LoRA fine-tuning** comparison
- **Rank sweep**: r = 2, 4, 8, 16
- **Metrics + plots + report** generated automatically

Target task: **SST-2** sentiment classification using a tiny BERT model (CPU-friendly).

## Setup

From the project root:

### macOS (Apple Silicon) / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows

```bash
cd lora_reproduction
# Use Python 3.11 (PyTorch-compatible on Windows)
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run experiments (full FT + LoRA rank sweep)

### macOS (Apple Silicon) / Linux

```bash
python -m src.train --config configs/sst2_bert_tiny.yaml --mode full
python -m src.train --config configs/sst2_bert_tiny.yaml --mode lora --lora_r 2
python -m src.train --config configs/sst2_bert_tiny.yaml --mode lora --lora_r 4
python -m src.train --config configs/sst2_bert_tiny.yaml --mode lora --lora_r 8
python -m src.train --config configs/sst2_bert_tiny.yaml --mode lora --lora_r 16

python -m src.evaluate --config configs/sst2_bert_tiny.yaml --runs_dir outputs/runs
python -m scripts.make_report --runs_dir outputs/runs
```

### Windows

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

### macOS (Apple Silicon) / Linux

```bash
python -m scripts.run_experiments --config configs/sst2_bert_tiny.yaml --ranks 2,4,8,16 --with_peft
```

### Windows

```bash
.\.venv\Scripts\python -m scripts.run_experiments --config configs\sst2_bert_tiny.yaml --ranks 2,4,8,16 --with_peft
```

Outputs:
- `outputs/runs/<run_name>/metrics.json`
- `outputs/metrics.csv`
- `outputs/plots/*.png`
- `results.md`

## Inference (single example)

### macOS (Apple Silicon) / Linux

```bash
python -m src.inference --run_dir outputs/runs/<run_name> --text "this movie was surprisingly good"
```

### Windows

```bash
py -3.11 -m src.inference --run_dir outputs\runs\<run_name> --text "this movie was surprisingly good"
```

## Notes

- Designed for **CPU-only**. Works on Windows and macOS (Apple Silicon).
- Uses small dataset subsets by default for speed; increase in `configs/*.yaml` if you have more time/RAM.


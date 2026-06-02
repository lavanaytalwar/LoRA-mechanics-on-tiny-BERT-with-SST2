# LoRA reproduction report (lightweight)

## What we reproduced

- Custom LoRA layer injection with frozen base weights (`src/lora.py`)
- Trainable low-rank adapters (A/B) + scaling alpha/r
- Full fine-tuning vs LoRA fine-tuning comparison
- Rank sweep: r = 2, 4, 8, 16
- Parameter-efficiency + checkpoint size + runtime measurements

## Environment constraints

- CPU-only, small model (`prajjwal1/bert-tiny`), SST-2 subset.

## Key results (best runs)

### Best full fine-tune

| mode   | lora_r   |   accuracy |     f1 |   trainable_params |   total_params |   train_seconds | checkpoint_bytes   |
|:-------|:---------|-----------:|-------:|-------------------:|---------------:|----------------:|:-------------------|
| full   |          |      0.696 | 0.7216 |          4,386,178 |      4,386,178 |            23.6 | 17.64 MB           |

### LoRA (custom) rank sweep

| mode   |   lora_r |   accuracy |     f1 |   trainable_params |   total_params |   train_seconds | checkpoint_bytes   |
|:-------|---------:|-----------:|-------:|-------------------:|---------------:|----------------:|:-------------------|
| lora   |        2 |      0.682 | 0.7028 |              9,986 |      4,395,906 |            26.4 | 17.68 MB           |
| lora   |        4 |      0.706 | 0.7351 |             19,714 |      4,405,634 |            28.4 | 17.71 MB           |
| lora   |        8 |      0.7   | 0.7024 |             39,170 |      4,425,090 |            36.3 | 17.79 MB           |
| lora   |       16 |      0.692 | 0.7061 |             78,082 |      4,464,002 |            33.1 | 17.94 MB           |

### LoRA (PEFT) rank sweep

| mode      |   lora_r |   accuracy |    f1 |   trainable_params |   total_params |   train_seconds | checkpoint_bytes   |
|:----------|---------:|-----------:|------:|-------------------:|---------------:|----------------:|:-------------------|
| peft_lora |        8 |      0.698 | 0.701 |             39,170 |      4,425,348 |            29.3 | 1.06 MB            |

## Observations vs paper expectations

- **Trainable parameter reduction**: LoRA trains a small fraction of parameters while keeping base weights frozen.
- **Quality trend vs rank**: accuracy typically improves with rank up to a point (diminishing returns).
- **Checkpoint size**: LoRA runs save smaller effective updates (in this repo we save full HF checkpoints for simplicity; see `Future improvements`).

## What differs (scaled-down reproduction)

- We reproduce mechanics + trends on a tiny encoder model and SST-2 subset, not large-scale generative LMs.
- We use short CPU-friendly runs and small data slices.

## Artifacts

- Metrics CSV: `outputs/metrics.csv`

- Plots: `outputs/plots`

## Future improvements

- Save **adapter-only** weights for custom LoRA (A/B matrices only) for true checkpoint-size comparisons.
- Add a tiny causal LM task (perplexity) to mirror the original paper setting.

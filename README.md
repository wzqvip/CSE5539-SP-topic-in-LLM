# CSE 5539 Homework: ModernBERT on SST-2

This folder contains the implementation for homework sections 4.1 and 4.2:

- head tuning: freeze the ModernBERT backbone and train only the classifier head;
- LoRA tuning: train a small LoRA adapter with a comparable trainable-parameter budget.

## Setup

Use Python 3.10 or newer. The commands below create an isolated environment and install the required packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CUDA machine, install the PyTorch build appropriate for the installed CUDA version from https://pytorch.org/get-started/locally/ before installing the remaining requirements.

## Run

Run a quick end-to-end smoke test first:

```powershell
python train_modernbert.py --mode both --epochs 1 --max-train-samples 128 --max-eval-samples 128 --output-dir runs/smoke
```

Run the full experiment (GPU recommended):

```powershell
python train_modernbert.py --mode both --epochs 3 --output-dir runs/modernbert-sst2
```

The script downloads `answerdotai/ModernBERT-base` and the GLUE SST-2 dataset from Hugging Face on first use. It writes the following files under the output directory:

- `accuracy_curves.png`: train/dev accuracy curves for both methods;
- `metrics.json`: best validation and test metrics;
- `parameter_counts.json`: total and trainable parameter counts;
- `table1.md`: the table to copy into the report;
- `head_tuning/` and `lora/`: best model checkpoints.

The script selects the checkpoint with the highest validation accuracy. SST-2's public test split may not expose labels; in that case the script records `test_accuracy` as `null` and the validation result remains the reproducible Table 1 metric.

The repository includes a successful small smoke run in `runs/smoke/`. Its numbers are only pipeline checks, not final homework results: head tuning reached 0.5000 validation accuracy and LoRA reached 0.5625 on 16 training and 16 validation examples.

## Experiment interpretation

Head tuning updates only the final classification layer. LoRA keeps the pretrained model frozen and injects low-rank trainable matrices into ModernBERT attention/MLP projections. The script reports trainable parameter counts so the LoRA rank can be adjusted to stay in the same order of magnitude as the head-only experiment.

See [TODO.md](TODO.md) for the remaining report and local-run checklist.


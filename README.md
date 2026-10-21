# Comparing Neural Models for AG News Topic Classification

## Project Overview

This project compares four neural network models for AG News topic classification:

- feed-forward neural network (FNN)
- recurrent neural network (RNN)
- long short-term memory network (LSTM)
- LSTM with attention

The final research question is:

> On AG News topic classification, does modeling word order with RNN, LSTM, or Attention LSTM improve performance enough to justify extra training cost compared with a simple averaged-embedding FNN, and how does maximum sequence length affect this trade-off?

## Repository Structure

- `src/`: main project code for data loading, models, training, and plotting.
- `reports/`: report files.
- `results/`: saved metrics, result summaries, figures, per-class metrics, and confusion matrices.
- `results/figures/`: generated plots.

## Environment Setup

The project was developed with Python 3.14.6.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data

The project uses the AG News dataset through the Hugging Face `datasets` library. The raw AG News dataset is not included in this repository. It is downloaded automatically on the first run and cached locally by Hugging Face after that.

No private data, password, or access token is needed.

Users should consult the AG News dataset/source card for current use restrictions. The submitted repository contains code, hyperparameters, saved metrics, plots, and analysis summaries, not the raw dataset files.

The code can generate prediction CSVs and Markdown error examples with `--save-analysis`. These files contain raw AG News text, so they are intentionally not included in the final submission. The included analysis artifacts are per-class metrics and confusion matrices.

## Main Experiment Reproduction

The main controlled comparison uses:

- epochs: 5
- batch size: 32
- max sequence length: 80
- max vocabulary size: 10000
- train/validation/test limits: 10000 / 2000 / 2000
- output file: `results/final_metrics.csv`

Run the four models with:

```bash
.venv/bin/python src/train.py --model fnn --epochs 5 --batch-size 32 --max-length 80 --max-vocab-size 10000 --train-limit 10000 --validation-limit 2000 --test-limit 2000 --output-path results/final_metrics.csv

.venv/bin/python src/train.py --model rnn --epochs 5 --batch-size 32 --max-length 80 --max-vocab-size 10000 --train-limit 10000 --validation-limit 2000 --test-limit 2000 --output-path results/final_metrics.csv --append-metrics

.venv/bin/python src/train.py --model lstm --epochs 5 --batch-size 32 --max-length 80 --max-vocab-size 10000 --train-limit 10000 --validation-limit 2000 --test-limit 2000 --output-path results/final_metrics.csv --append-metrics

.venv/bin/python src/train.py --model attention --epochs 5 --batch-size 32 --max-length 80 --max-vocab-size 10000 --train-limit 10000 --validation-limit 2000 --test-limit 2000 --output-path results/final_metrics.csv --append-metrics
```

## Sequence Length Experiment Reproduction

The sequence-length analysis is saved in `results/sequence_length_metrics.csv`.

It uses:

- models: `fnn`, `lstm`, `attention`
- max sequence length: 30
- epochs: 4
- batch size: 32
- max vocabulary size: 10000
- train/validation/test limits: 10000 / 2000 / 2000

To reproduce it, run `src/train.py` for each model and each max length. Use a fresh output file for the first run and `--append-metrics` for the remaining runs. For example:

```bash
.venv/bin/python src/train.py --model fnn --epochs 4 --batch-size 32 --max-length 30 --max-vocab-size 10000 --train-limit 10000 --validation-limit 2000 --test-limit 2000 --output-path results/sequence_length_metrics.csv

.venv/bin/python src/train.py --model lstm --epochs 4 --batch-size 32 --max-length 30 --max-vocab-size 10000 --train-limit 10000 --validation-limit 2000 --test-limit 2000 --output-path results/sequence_length_metrics.csv --append-metrics
```

Repeat the same pattern for `attention` and for max lengths `80` and `150`.

## Evaluation and Plots

Generate the main comparison plots with:

```bash
.venv/bin/python src/evaluate.py --metrics-path results/final_metrics.csv --output-dir results/figures
```

Generated figures include:

- `results/figures/validation_accuracy.png`
- `results/figures/validation_loss.png`
- `results/figures/final_test_accuracy.png`
- `results/figures/training_time.png`
- `results/figures/sequence_length_test_accuracy.png`

## Important Result Files

- `results/final_metrics.csv`
- `results/final_analysis_summary.md`
- `results/sequence_length_summary.md`
- `results/figures/`
- `results/analysis/*/*_class_metrics.csv`
- `results/analysis/*/*_confusion_matrix.csv`
- `results/analysis/*/*_confusion_matrix.png`

## Reproducibility Notes

- The random seed is controlled with `--seed`.
- The default seed is `42`.
- Exact results may vary slightly depending on hardware and backend.
- The recorded experiments were run on macOS using MPS.

## Final Submission Note

The final submission should include:

- `README.md`
- `src/`
- `results/`
- `reports/report.pdf`
- `requirements.txt`

The final submission should not include `.venv/`, cache files, private data, unnecessary course materials, raw prediction CSVs, or raw error-example Markdown files.

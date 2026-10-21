# Final Detailed Model Analysis

These are the final detailed evaluation results for the main controlled model comparison.

## Experiment Settings

- models: `fnn`, `rnn`, `lstm`, `attention`
- epochs: 5
- batch size: 32
- max length: 80
- max vocab size: 10000
- train limit: 10000
- validation limit: 2000
- test limit: 2000
- seed: 42
- metrics file: `results/final_metrics.csv`
- analysis files: `results/analysis/`
- checkpoint saving: no

## Final Metrics

| Model | Validation accuracy | Test accuracy | Test macro precision | Test macro recall | Test macro F1 | Parameter count | Training time seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FNN | 0.8405 | 0.8460 | 0.8452 | 0.8441 | 0.8442 | 644420 | 7.64 |
| RNN | 0.7070 | 0.6775 | 0.6821 | 0.6698 | 0.6682 | 648580 | 29.02 |
| LSTM | 0.8100 | 0.7940 | 0.7952 | 0.7927 | 0.7928 | 673540 | 17.24 |
| Attention LSTM | 0.8125 | 0.8210 | 0.8211 | 0.8188 | 0.8195 | 673605 | 18.95 |

## Model Comparison

The FNN achieved the best validation accuracy, test accuracy, and macro F1. This supports the interpretation that AG News topic classification is strongly driven by topic words, so an averaged-embedding model can perform very well without explicitly modeling word order.

The Attention LSTM was the strongest sequence model and clearly outperformed the plain LSTM on test accuracy and macro F1. The plain RNN was weaker than the other models.

## Computational Cost

The FNN was also the fastest model in this run. The RNN was the slowest despite having the weakest test performance. The LSTM and Attention LSTM required substantially more training time and had more parameters, but only the Attention LSTM came close to the FNN's accuracy.

## Detailed Analysis Files

Per-class metrics and confusion matrices are saved under `results/analysis/`.

- `results/analysis/fnn/`
- `results/analysis/rnn/`
- `results/analysis/lstm/`
- `results/analysis/attention/`

Each model directory in the final submission contains:

- per-class metrics CSV
- confusion matrix CSV
- confusion matrix PNG

Raw-text prediction CSVs and full error-example Markdown files can be regenerated locally with `--save-analysis`, but they are excluded from the submitted package because they contain AG News text.

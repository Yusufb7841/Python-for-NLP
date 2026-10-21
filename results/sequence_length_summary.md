# Sequence Length Analysis

This is an additional analysis, not a replacement for the main controlled comparison.

## Experiment Question

How does the maximum sequence length affect model performance?

## Settings

- models: `fnn`, `lstm`, `attention`
- max sequence lengths: 30, 80, 150
- epochs: 4
- batch size: 32
- max vocab size: 10000
- train limit: 10000
- validation limit: 2000
- test limit: 2000
- metrics file: `results/sequence_length_metrics.csv`
- the metrics file includes a `max_length` column for each run
- checkpoint saving: no

## Final Metrics

| Model | Max length | Final validation accuracy | Final test accuracy | Training time seconds |
| --- | ---: | ---: | ---: | ---: |
| FNN | 30 | 0.8070 | 0.8190 | 7.43 |
| LSTM | 30 | 0.7915 | 0.7845 | 10.25 |
| Attention LSTM | 30 | 0.7825 | 0.7795 | 9.90 |
| FNN | 80 | 0.8405 | 0.8355 | 7.78 |
| LSTM | 80 | 0.7920 | 0.7855 | 14.69 |
| Attention LSTM | 80 | 0.8115 | 0.8095 | 15.22 |
| FNN | 150 | 0.8435 | 0.8350 | 10.21 |
| LSTM | 150 | 0.8040 | 0.7780 | 19.63 |
| Attention LSTM | 150 | 0.8150 | 0.8090 | 20.16 |

## Interpretation

The FNN improved when max length increased from 30 to 80, but there was almost no extra test accuracy gain from 80 to 150. The attention LSTM also improved from 30 to 80 and stayed almost the same at 150. The plain LSTM changed only slightly across the three lengths.

For this setup, max length 80 looks like a reasonable choice. It captures more information than 30 without adding as much training time as 150.

Generated plot: `results/figures/sequence_length_test_accuracy.png`

# Does word order matter for news classification?

[![CI](https://github.com/Yusufb7841/word-order-text-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/Yusufb7841/word-order-text-classification/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
[![Report](https://img.shields.io/badge/report-PDF-informational)](reports/report.pdf)

RNNs and LSTMs were built to read text in order. Do they need that to sort news into
topics? This project pits four neural models, from an order-blind bag of embeddings to an
attention LSTM, against a TF-IDF baseline on AG News, in 180 training runs with three seeds
each. **Short answer: not here.** A model that ignores word order matches the LSTM at a
sixth of the training cost.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/main_results-dark.png">
  <img alt="Test accuracy and estimated CPU training time per model" src="results/figures/main_results.png">
</picture>

## Key findings

- **Word order barely helps.** The GloVe LSTM (92.2%), an order-blind FNN (92.0%) and
  TF-IDF + logistic regression (92.0%) are tied within seed-to-seed noise.
- **Pretrained embeddings matter more than architecture.** With 1,000 training examples,
  GloVe lifts the LSTM from 60.8% to 82.5%.
- **The headline is enough.** The first 8 tokens alone give 85–88% accuracy.
- **The LSTMs cost about 6× more to train** than the FNN (with GloVe), for no gain in
  accuracy.

The [full write-up (PDF)](reports/report.pdf) covers the method, every result and the
limitations.

## More results

### Do the sequence models just need more data?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/learning_curve-dark.png">
  <img alt="Test accuracy versus number of training examples" src="results/figures/learning_curve.png">
</picture>

With little data, the order-blind FNN wins clearly: at 1,000 examples the GloVe FNN reaches
85.3%, against 82.5% for both TF-IDF and the GloVe LSTM. With GloVe, the LSTM catches up
from about 30,000 examples; with random embeddings it still trails the FNN on the full data.

<details>
<summary>Table: accuracy by training-set size</summary>

<!-- BEGIN table:learning_curve -->
| Model | Embeddings | 1,000 | 3,000 | 10,000 | 30,000 | 108,000 (all) |
| :----------------- | :---------- | -----: | -----: | ------: | ------: | -------: |
| TF-IDF + LogReg | n/a | 82.5 | 86.2 | 88.6 | 90.2 | 92.0 |
| FNN (mean-pooled) | GloVe | 85.3 | 87.7 | 89.3 | 90.7 | 92.0 |
| FNN (mean-pooled) | random | 74.3 | 81.6 | 86.2 | 89.1 | 91.6 |
| RNN | GloVe | 78.2 | 84.1 | 86.4 | 87.8 | 89.3 |
| RNN | random | 43.8 | 63.9 | 78.5 | 84.2 | 89.1 |
| LSTM | GloVe | 82.5 | 87.0 | 89.1 | 91.1 | 92.2 |
| LSTM | random | 60.8 | 74.5 | 82.5 | 87.5 | 90.9 |
| Attention LSTM | GloVe | 81.5 | 87.0 | 89.1 | 90.7 | 92.1 |
| Attention LSTM | random | 62.8 | 73.6 | 81.9 | 87.1 | 90.5 |
<!-- END table:learning_curve -->

Test accuracy (%), mean over 3 seeds.

</details>

### How much of each article is needed?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/sequence_length-dark.png">
  <img alt="Test accuracy versus number of tokens kept per article" src="results/figures/sequence_length.png">
</picture>

Every model is retrained on only the first 8, 16, 32 or 64 tokens. The first 8, roughly
the headline, already give 85–88%, and the ranking of the models is the same at every
length.

<details>
<summary>Table: accuracy by input length</summary>

<!-- BEGIN table:sequence_length -->
| Model | 8 tokens | 16 tokens | 32 tokens | 64 tokens |
| :----------------- | ------: | ------: | ------: | ------: |
| TF-IDF + LogReg | 87.8 | 90.0 | 91.7 | 92.0 |
| FNN (mean-pooled) | 87.1 | 89.4 | 91.3 | 91.6 |
| RNN | 85.8 | 88.3 | 89.0 | 89.1 |
| LSTM | 85.4 | 88.5 | 90.4 | 90.9 |
| Attention LSTM | 86.1 | 88.3 | 90.3 | 90.5 |
<!-- END table:sequence_length -->

Test accuracy (%), random-init embeddings, mean over 3 seeds. Articles have a median of
38 tokens; only 2% are longer than 64.

</details>

### What does the attention layer look at?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/attention_examples-dark.png">
  <img alt="Test articles with each token shaded by its attention weight" src="results/figures/attention_examples.png">
</picture>

The weights often land on topical words (*phelps*, *medal*, *phishing*, *fraud*). Each
LSTM state has already read everything to its left, though, so they show where the model
pools information, not which word decided the prediction.

## Quick start

```bash
git clone https://github.com/Yusufb7841/word-order-text-classification.git && cd word-order-text-classification
make setup     # .venv with pinned dependencies
make test      # unit tests on synthetic data, ~5 s
make results   # GloVe download, all 180 runs, timing benchmark, figures, tables, PDF
```

The full sweep is about 4.5 hours of training on an Apple M2 laptop. It is resumable:
finished runs are recorded in `results/runs.csv` and skipped on restart.

## Details

<details>
<summary><b>Full results table and training cost</b></summary>

<!-- BEGIN table:main -->
| Model | Embeddings | Test accuracy (%) | Macro F1 (%) | Best epoch | CPU train time (min) | Params excl. embeddings |
| :----------------- | :---------- | ------------: | ------------: | -----: | -----: | ----------: |
| TF-IDF + LogReg | n/a | 92.04 ± 0.07 | 92.02 ± 0.07 | n/a | 0.9 | n/a |
| FNN (mean-pooled) | GloVe | 92.02 ± 0.11 | 92.01 ± 0.11 | 4.3 | 0.6 | 13,444 |
| FNN (mean-pooled) | random | 91.60 ± 0.11 | 91.59 ± 0.11 | 8.0 | 0.9 | 13,444 |
| RNN | GloVe | 89.28 ± 0.48 | 89.26 ± 0.48 | 4.7 | 1.6 | 29,956 |
| RNN | random | 89.09 ± 0.44 | 89.07 ± 0.45 | 8.7 | 2.5 | 29,956 |
| LSTM | GloVe | 92.21 ± 0.23 | 92.19 ± 0.24 | 3.0 | 3.6 | 118,276 |
| LSTM | random | 90.91 ± 0.30 | 90.90 ± 0.29 | 4.3 | 4.4 | 118,276 |
| Attention LSTM | GloVe | 92.12 ± 0.22 | 92.11 ± 0.21 | 2.7 | 3.7 | 134,916 |
| Attention LSTM | random | 90.55 ± 0.21 | 90.53 ± 0.19 | 3.7 | 4.3 | 134,916 |
<!-- END table:main -->

Mean ± standard deviation over 3 seeds on the full 7,600-article test set (one standard
error of an accuracy near 92% is about 0.3 points). *CPU train time* is the controlled
per-epoch time below multiplied by the epochs each configuration actually ran.

Seconds per epoch over the full 108k-example training split, one model at a time on an
otherwise idle Apple M2 (8 cores, 16 GB, PyTorch 2.15, default 4 CPU threads):

<!-- BEGIN table:timing -->
| Model | CPU s/epoch | MPS s/epoch |
| :----------------- | -------: | -------: |
| FNN (mean-pooled) | 5.1 | 6.2 |
| RNN | 12.8 | 28.2 |
| LSTM | 35.9 | 18.6 |
| Attention LSTM | 39.1 | 21.1 |

TF-IDF + LogReg: 56 s for a full CPU fit (including the C search).
<!-- END table:timing -->

On Apple's MPS backend the LSTM runs about twice as fast as on the CPU and the plain RNN
about twice as slow. The original version of this project reported "the RNN is slower than
the LSTM" as a property of the models; it is a property of the backend.

</details>

<details>
<summary><b>Per-class F1 and confusion matrices</b></summary>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/confusion_matrices-dark.png">
  <img alt="Row-normalized confusion matrices for each model" src="results/figures/confusion_matrices.png">
</picture>

<!-- BEGIN table:per_class -->
| Model | Embeddings | World | Sports | Business | Sci/Tech |
| :----------------- | :---------- | -----: | ------: | --------: | --------: |
| TF-IDF + LogReg | n/a | 92.2 | 96.6 | 89.3 | 90.0 |
| FNN (mean-pooled) | GloVe | 92.2 | 97.1 | 88.8 | 89.9 |
| FNN (mean-pooled) | random | 91.8 | 96.9 | 88.3 | 89.3 |
| RNN | GloVe | 88.9 | 95.7 | 85.3 | 87.1 |
| RNN | random | 89.4 | 95.3 | 84.9 | 86.7 |
| LSTM | GloVe | 92.3 | 97.3 | 89.0 | 90.1 |
| LSTM | random | 91.3 | 96.1 | 87.8 | 88.5 |
| Attention LSTM | GloVe | 92.4 | 97.2 | 88.9 | 89.8 |
| Attention LSTM | random | 90.9 | 95.7 | 87.2 | 88.4 |
<!-- END table:per_class -->

Per-class F1 (%), mean over seeds.

</details>

<details>
<summary><b>Method</b></summary>

| | |
|:--|:--|
| **Data** | AG News via Hugging Face `datasets` (downloaded on first run, not redistributed here). The official 120k training set is split 90/10 (stratified, per seed) into train and validation; the official 7.6k test set is used only for final evaluation. |
| **Preprocessing** | AG News' broken HTML entities (`#39;`, `quot;`, `&lt;b&gt;`) are repaired, then text is lowercased and split into word/number tokens. Vocabulary: the 20k most frequent training tokens seen at least twice. |
| **Models** | Shared 100-d embeddings and 128-d hidden size. FNN = masked mean of embeddings → MLP. RNN / LSTM = hidden state at the last real token. Attention LSTM = additive attention pooling over all LSTM states. Dropout 0.2 before the classifier. |
| **Embeddings** | Random init, or initialized from [GloVe 6B 100d](https://nlp.stanford.edu/projects/glove/) (99% vocabulary coverage) and fine-tuned. |
| **Baseline** | TF-IDF over unigrams + bigrams (sublinear TF, min_df 2) with logistic regression, C chosen on validation. |
| **Training** | Adam (lr 1e-3), batch 64, gradient clipping at 1.0, up to 30 epochs. Early stopping on validation accuracy with patience 3; the best-validation weights are evaluated on test. |
| **Evaluation** | Accuracy and macro F1 on the full test set, mean ± std over seeds 0, 1, 2. Learning-curve subsets are stratified samples of the training split; the validation set stays fixed. |

Hyperparameters are shared by every model (see [`src/agnews/config.py`](src/agnews/config.py))
and were not tuned per model.

</details>

<details>
<summary><b>Running parts of the sweep</b></summary>

```bash
python -m agnews.experiments --suites main --seeds 0      # one suite / seed
python -m agnews.experiments --models fnn tfidf --dry-run  # list pending runs
python -m agnews.benchmark                                 # timing table
python -m agnews.figures && python -m agnews.attention_viz
python -m agnews.summary                                   # refresh tables in this README
```

Several worker processes can share the sweep: give them disjoint `--models` lists, or
let a second worker take the same queue from the other end with `--reverse` (appends to
the results file are locked). The results here came from three workers on one laptop,
with LSTM-based models mostly on Apple's MPS backend and everything else on CPU; the
device used for each run is recorded in `results/runs.csv`.

</details>

<details>
<summary><b>Project structure</b></summary>

```
src/agnews/
  data.py           cleaning, tokenization, vocabulary, stratified splits, batching
  models.py         FNN, RNN/LSTM, Attention LSTM
  embeddings.py     GloVe download + embedding matrix
  baselines.py      TF-IDF + logistic regression
  training.py       training loop, early stopping, prediction
  experiments.py    experiment suites -> results/runs.csv, results/history.csv
  benchmark.py      controlled timing -> results/timing.csv
  figures.py        all plots (light + dark)
  attention_viz.py  attention heatmaps
  summary.py        tables -> results/summary.md, README.md, reports/report.md
tests/              pytest suite (runs in CI on every push)
reports/            write-up (report.md -> report.pdf)
```

</details>

<details>
<summary><b>What changed from the original course version</b></summary>

This project started as a course assignment built in about three hours
([original version](https://github.com/Yusufb7841/word-order-text-classification/tree/v1.0-course)). Reviewing it afterwards turned up problems
that changed its conclusions, so it was rebuilt:

| v1 (course submission) | v2 (this version) |
|:--|:--|
| 10k of 120k training examples, 2k of 7.6k test examples | Full training and test sets, plus a learning curve over 1k–108k |
| One seed; differences of 1–3 points reported as "clearly better" | 3 seeds, mean ± std; with 2k test examples the old 95% interval was already ±1.7 points |
| Sequence lengths 30 / 80 / 150. Only 0.5% of articles exceed 80 tokens and none exceed 150, so 80 vs 150 compared identical inputs | Lengths 8 / 16 / 32 / 64, chosen from the length distribution |
| Whitespace tokenization kept `#39;s`, `quot;`, `(reuters)` and `said.` as separate words | Entity repair + word tokenizer |
| RNN confusion matrix from a different run than the reported metrics (63.8% vs 67.8% accuracy) | Every table and figure is generated from one results file |
| "RNN is slower than LSTM", an artifact of PyTorch's MPS backend | Controlled timing on CPU and MPS reported separately |
| Parameter counts compared, but 99% of them were the shared embedding table | Non-embedding parameters reported |
| No baseline outside the four neural models | TF-IDF + logistic regression and GloVe embeddings |
| No tests, hand-run commands | pytest + CI, one-command reproduction |

</details>

<details>
<summary><b>Limitations</b></summary>

- **One dataset.** AG News is a topic task with strong keyword cues. The conclusions are
  not expected to carry over to tasks where word order carries meaning, such as negation,
  sentiment or entailment.
- **Shared hyperparameters.** All models use one configuration, not tuned per model.
- **No pretrained Transformer.** A fine-tuned BERT-style model (about 95% on AG News in
  published results) would be the natural upper bound and next step.

</details>

<details>
<summary><b>References</b></summary>

- Zhang, Zhao & LeCun (2015). *Character-level Convolutional Networks for Text Classification.* NeurIPS. (AG News benchmark)
- Pennington, Socher & Manning (2014). *GloVe: Global Vectors for Word Representation.* EMNLP.
- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory.* Neural Computation.
- Bahdanau, Cho & Bengio (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR.
- Joulin et al. (2017). *Bag of Tricks for Efficient Text Classification.* EACL. (fastText)
- Jain & Wallace (2019). *Attention is not Explanation.* NAACL.

</details>

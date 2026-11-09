# Does word order matter for news topic classification?

[![CI](https://github.com/Yusufb7841/Python-for-NLP/actions/workflows/ci.yml/badge.svg)](https://github.com/Yusufb7841/Python-for-NLP/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)

A controlled comparison of four neural text classifiers — a mean-pooled embedding **FNN**
that ignores word order, a vanilla **RNN**, an **LSTM**, and an **LSTM with attention** —
against a classic **TF-IDF + logistic regression** baseline on the AG News topic
classification benchmark (4 classes, 120k training / 7.6k test articles).

Every configuration is trained with 3 seeds on the full dataset, with random and
GloVe-initialized embeddings, across 5 training-set sizes and 4 input lengths
(180 training runs). Every number and figure on this page is produced by a script from
[`results/runs.csv`](results/runs.csv).

<!-- FINDINGS -->

## Results

### Main comparison

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/main_results-dark.png">
  <img alt="Test accuracy and estimated CPU training time per model" src="results/figures/main_results.png">
</picture>

<!-- BEGIN table:main -->
<!-- END table:main -->

Mean ± standard deviation over 3 seeds on the full 7,600-article test set (at this test
size, one standard error of an accuracy near 92% is about 0.3 points). *CPU train time* is
the controlled per-epoch time from the timing benchmark multiplied by the number of epochs
each configuration actually ran, including the 3 early-stopping patience epochs.

### Do the sequence models just need more data?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/learning_curve-dark.png">
  <img alt="Test accuracy versus number of training examples" src="results/figures/learning_curve.png">
</picture>

<!-- BEGIN table:learning_curve -->
<!-- END table:learning_curve -->

### How much of each article is needed?

AG News articles are short: a median of 38 tokens, with only 2% longer than 64. To find
out where the signal sits, every model is retrained seeing only the first 8, 16, 32 or
64 tokens.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/sequence_length-dark.png">
  <img alt="Test accuracy versus number of tokens kept per article" src="results/figures/sequence_length.png">
</picture>

<!-- BEGIN table:sequence_length -->
<!-- END table:sequence_length -->

### Where the models go wrong

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/confusion_matrices-dark.png">
  <img alt="Row-normalized confusion matrices for each model" src="results/figures/confusion_matrices.png">
</picture>

Per-class F1 (%), mean over seeds:

<!-- BEGIN table:per_class -->
<!-- END table:per_class -->

### What does the attention layer look at?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/attention_examples-dark.png">
  <img alt="Test articles with each token shaded by its attention weight" src="results/figures/attention_examples.png">
</picture>

The weights often land on topical words (*phelps*, *medal*, *phishing*, *fraud*). But
across the test set, the GloVe model puts 19.3% of its attention on the first 8 tokens
(roughly the headline) and 24.7% on the last 8, where uniform attention would give 22.4%
to each ([`results/attention_stats.txt`](results/attention_stats.txt)). Each LSTM state
summarizes everything to its left, so late states have already read the headline. The
weights show where the model pools information, not which word caused the prediction.

### Training cost

Seconds per training epoch over the full 108k-example training split, measured one model
at a time on an Apple M2 laptop (8 cores, 16 GB, PyTorch 2.15):

<!-- BEGIN table:timing -->
<!-- END table:timing -->

<!-- TIMING -->

## Method

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

## Reproducing

```bash
git clone https://github.com/Yusufb7841/Python-for-NLP.git && cd Python-for-NLP
make setup        # .venv with pinned dependencies (or: pip install -r requirements.txt)
make test         # unit tests, synthetic data only, ~5 s
make results      # GloVe download, all 180 runs, timing benchmark, figures, tables, PDF
```

`make results` takes about 2.5 hours on an Apple M2. The sweep is resumable: finished
runs are recorded in `results/runs.csv` and skipped on restart. Individual steps:

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

## Project structure

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

## What changed from the original version

This project started as a course assignment built in about three hours
([original version](https://github.com/Yusufb7841/Python-for-NLP/tree/v1.0-course)).
Reviewing it afterwards turned up problems that changed its conclusions, so it was
rebuilt:

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

## Limitations

- **Single dataset.** AG News is a topic task with strong keyword cues, so the conclusions
  are not expected to transfer to tasks where word order carries meaning (negation,
  sentiment, entailment).
- **Fixed hyperparameters.** All models share one configuration. A per-model search might
  narrow or widen the gaps, though the gaps are small either way.
- **No pretrained Transformer.** A fine-tuned BERT-style model would be the natural
  upper bound (about 95% on AG News in published results) and the obvious next step.
- **Attention is not explanation.** The attention weights apply to LSTM states that already
  mix in earlier words, so the heatmaps show where the model pools information, not which
  word caused a decision.

## References

- Zhang, Zhao & LeCun (2015). *Character-level Convolutional Networks for Text Classification.* NeurIPS. (AG News benchmark)
- Pennington, Socher & Manning (2014). *GloVe: Global Vectors for Word Representation.* EMNLP.
- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory.* Neural Computation.
- Bahdanau, Cho & Bengio (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR.
- Joulin et al. (2017). *Bag of Tricks for Efficient Text Classification.* EACL. (fastText)
- Jain & Wallace (2019). *Attention is not Explanation.* NAACL.

The full write-up is in [`reports/report.pdf`](reports/report.pdf).

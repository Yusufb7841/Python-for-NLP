---
title: "Does Word Order Matter for News Topic Classification?"
subtitle: "A controlled comparison of bag-of-words, recurrent and attention models on AG News"
author: "Yusuf Bulut"
date: "November 2026 (revised edition of a course project)"
geometry: margin=2.3cm
fontsize: 11pt
mainfont: "Charter"
sansfont: "Helvetica Neue"
monofont: "Menlo"
linkcolor: blue
urlcolor: blue
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \floatplacement{figure}{H}
  - \renewcommand{\arraystretch}{1.15}
---

<!-- ABSTRACT -->

# 1 Introduction

Recurrent networks and attention were designed to model word order, and they are standard
tools for text classification. Topic classification, however, can often be solved from
which words occur, regardless of their order. A story mentioning *touchdown* and
*quarterback* is about sports wherever those words appear. Bag-of-words models
are known to be strong on such tasks [@zhang2015; @joulin2017; @iyyer2015]. This project
asks how much modeling word order actually adds on a standard topic benchmark, and what it
costs:

> On AG News, do models that read word order (RNN, LSTM, Attention LSTM) classify topics
> more accurately than an order-blind model that averages word embeddings, and does the
> answer depend on the amount of training data, the input length or the initialization of
> the embeddings?

The comparison is controlled: all neural models share their embedding size, hidden size,
vocabulary, optimizer and stopping rule, and differ only in how they turn a sequence of
word vectors into one document vector. A TF-IDF + logistic regression baseline anchors the
results, and every configuration is repeated with three seeds so that differences can be
compared with the noise.

# 2 Data

AG News [@zhang2015] contains news articles from four topics (World, Sports, Business,
Sci/Tech), each consisting of a headline and a short description. The official split has
120,000 training and 7,600 test articles, balanced across classes. The dataset is loaded
through Hugging Face `datasets` and is not redistributed with this project; its source
describes it as provided for research and non-commercial use.

For every seed, the official training set is split 90/10 into training (108,000) and
validation (12,000) sets, stratified by class. The test set is used only once per run, for
the final evaluation of the weights with the best validation accuracy.

**Preprocessing.** AG News stores many HTML entities without their leading ampersand
(`#39;s`, `quot;`), uses backslashes as line breaks, and occasionally contains escaped HTML
tags. These are repaired before tokenization; the original version of this project skipped
this step, which put tokens such as `#39;s` and `(reuters)` into the vocabulary. Text is
then lowercased and split into alphanumeric tokens (dotted forms such as `u.s` are kept).
After cleaning, articles have a median length of 38 tokens; 98% have 64 tokens or fewer,
and none exceed 181.

# 3 Models

All neural models share the same embedding layer (vocabulary of the 20,000 most frequent
training tokens appearing at least twice, 100 dimensions) and the same hidden size (128).
They differ only in how a sequence of word vectors becomes a single document vector:

- **FNN (mean-pooled).** The masked mean of the word embeddings, followed by a
  one-hidden-layer MLP. This model is invariant to word order.
- **RNN.** A single-layer Elman RNN [@elman1990]; the hidden state at the last real
  (non-padding) token is classified.
- **LSTM.** As above with an LSTM cell [@hochreiter1997].
- **Attention LSTM.** An LSTM whose states are pooled with additive attention
  [@bahdanau2015]: a two-layer scorer assigns each position a weight, padding is masked, and
  the weighted sum of states is classified.

Each model is trained with randomly initialized embeddings and, separately, with
embeddings initialized from GloVe 6B 100d [@pennington2014] and fine-tuned. GloVe covers
99% of the vocabulary.

**Baseline.** TF-IDF features over word unigrams and bigrams (sublinear term frequency,
minimum document frequency 2) with multinomial logistic regression. The inverse
regularization strength $C \in \{1, 4, 16\}$ is chosen on the validation set, the same data
the neural models use for early stopping.

# 4 Experimental setup

| Setting | Value |
|:--|:--|
| Optimizer | Adam, learning rate $10^{-3}$, batch size 64, gradient-norm clipping at 1.0 |
| Regularization | dropout 0.2 before the classifier |
| Stopping | up to 30 epochs; early stopping on validation accuracy, patience 3 |
| Input length | first 64 tokens (main experiment) |
| Seeds | 0, 1, 2. The seed controls the train/validation split, subsampling, initialization and batch order |
| Metrics | accuracy and macro F1 on all 7,600 test articles; mean $\pm$ standard deviation over seeds |

Hyperparameters are shared by all models and were not tuned per model. Three experiments
are run:

1. **Main comparison.** All models on the full training set, both embedding initializations.
2. **Learning curve.** Stratified training subsets of 1k, 3k, 10k and 30k articles, plus
   the full 108k, with the validation set held fixed.
3. **Input length.** Models trained and evaluated on only the first 8, 16, 32 or 64 tokens
   of each article (random-init embeddings).

All 180 runs are logged to a single results file from which every table and figure in this
report is generated. Training was run on an Apple M2 laptop by three concurrent workers:
LSTM-based models mostly on the MPS GPU backend and everything else on CPU (the device of
every run is logged). Because the workers shared the machine, training cost is measured
separately in a controlled benchmark.

# 5 Results

## 5.1 Main comparison

![Test accuracy (left) and estimated CPU training time (right) on the full training set. Hollow markers: random-init embeddings; filled: GloVe-init. Error bars: $\pm$ 1 standard deviation over seeds.](results/figures/main_results.png){width=100%}

<!-- BEGIN table:main -->
<!-- END table:main -->

: Main results (full training set, first 64 tokens). Mean $\pm$ standard deviation over three seeds. *Best epoch* is the epoch whose weights were kept. *CPU train time* is the controlled CPU time per epoch multiplied by the number of epochs actually run, including patience.

With GloVe initialization, three neural models reach the accuracy of the TF-IDF baseline
(92.04%): the order-blind FNN (92.02%), the LSTM (92.21%) and the Attention LSTM (92.12%).
These differences are small relative to the noise. The standard error of a single
accuracy measurement on 7,600 test articles is about 0.3 points, and the seed-to-seed
standard deviation of the LSTM is 0.23. Because the seed fixes the train/validation split,
the models can also be compared seed by seed against TF-IDF. The paired differences of the
GloVe FNN ($-0.21$ to $+0.14$) and GloVe Attention LSTM ($-0.09$ to $+0.20$) change sign
across seeds. Only the GloVe LSTM is ahead on two seeds out of three, and by at most 0.3
points.

The embedding initialization matters more than the architecture. GloVe improves the LSTM
by 1.3 points and the Attention LSTM by 1.6, but the FNN by only 0.4. With random
embeddings, the order-blind FNN (91.60%) outperforms the LSTM (90.91%) and the Attention
LSTM (90.55%). A recurrent model has to learn word representations and how to compose
them from the same labels, while a mean of embeddings only has to learn which words
indicate which topic. Attention never improves on the plain LSTM, and the vanilla RNN
trails every other model by about three points with either initialization.

## 5.2 Learning curve

![Test accuracy against training-set size (log scale) for random-init (left) and GloVe-init (right) embeddings. Shaded bands: $\pm$ 1 standard deviation over seeds.](results/figures/learning_curve.png){width=100%}

<!-- BEGIN table:learning_curve -->
<!-- END table:learning_curve -->

: Test accuracy (%) by number of training examples, mean over seeds.

<!-- LEARNING_CURVE -->

## 5.3 Input length

![Test accuracy when each article is truncated to its first 8, 16, 32 or 64 tokens (random-init embeddings). Tick labels give the share of training articles that are cut at each length.](results/figures/sequence_length.png){width=72%}

<!-- BEGIN table:sequence_length -->
<!-- END table:sequence_length -->

: Test accuracy (%) by number of tokens kept, mean over seeds.

<!-- SEQUENCE_LENGTH -->

## 5.4 Error analysis

![Row-normalized confusion matrices (percent of each true topic), summed over seeds, for the best embedding setting of each model.](results/figures/confusion_matrices.png){width=100%}

<!-- BEGIN table:per_class -->
<!-- END table:per_class -->

: Per-class F1 (%), mean over seeds.

All models agree on which topics are hard. Sports is the easiest (F1 above 95% for every
model) and Business the hardest, followed by Sci/Tech. For every model, confusions between
Business and Sci/Tech make up 41–46% of all errors; an even split over the six possible
pairs would give each about 17%. The next most frequent pair, World and Business, accounts
for about 20%. Technology-company stories use the vocabulary of both topics (earnings,
acquisitions, products), and some misclassified articles, such as the two at the bottom
of Figure 5, could plausibly belong to either class. Among the strongest models, the
per-class differences are a few tenths of a point. TF-IDF is slightly ahead on Business
and slightly behind on Sports, so no architecture has a consistent advantage on any one
topic.

## 5.5 What the attention layer attends to

![Six test articles shaded by the attention weights of the GloVe-initialized Attention LSTM (seed 0). The last two are misclassified.](results/figures/attention_examples.png){width=100%}

The attention weights often land on topical words: *phelps* and *medal*, *phishing*,
*fraud* and *cards*. Across the whole test set, however, the attention does not
concentrate on the opening tokens, which roughly correspond to the headline. The first 8
tokens receive 19.3% of the attention mass on average and the last 8 tokens 24.7%, where
uniform attention would give 22.4% to each. The random-init model leans further towards
the end (15.2% and 24.8%; `results/attention_stats.txt`).

This does not mean the headline is uninformative; Section 5.3 shows that the first 8
tokens alone support about 80% accuracy (TODO verify). An LSTM state at position $t$
summarizes every token up to $t$, so a late state has already read the headline. Pooling
late states is therefore an efficient way to read the whole article. The weights show
where the model pools information, not which word caused a prediction
[@jain2019; @wiegreffe2019].

## 5.6 Training cost

<!-- BEGIN table:timing -->
<!-- END table:timing -->

: Training time per epoch on the full training split (108,000 articles), measured one model at a time on an Apple M2.

<!-- TIMING -->


# 6 Discussion

<!-- DISCUSSION -->

# 7 Limitations

<!-- LIMITATIONS -->

# 8 Conclusion

<!-- CONCLUSION -->

# Reproducibility

Code, results and instructions are at
<https://github.com/Yusufb7841/Python-for-NLP>. `make results` reruns every experiment,
regenerates every figure and table, and rebuilds this PDF. Unit tests run in continuous
integration on every push.

# Changes from the original course version

The first version of this project was a course submission built in about three hours. A
later review found several problems serious enough to change its conclusions, and this
edition was rebuilt to fix them:

- **Data.** v1 trained on 10,000 of the 120,000 training articles and tested on 2,000 of
  the 7,600 test articles. All data is now used, and the effect of training-set size is
  measured explicitly.
- **Statistical support.** v1 reported a single seed. With 2,000 test articles, the 95%
  confidence interval of an accuracy near 82% is about $\pm$ 1.7 points, so several of
  its "clear" differences were within noise. All results are now means over three seeds,
  reported with standard deviations.
- **A sequence-length experiment that could not show anything.** v1 compared maximum
  lengths of 30, 80 and 150 tokens. Only 0.5% of articles exceed 80 whitespace tokens and
  none exceed 150, so the 80 and 150 conditions saw identical inputs. The new lengths
  (8–64) were chosen from the measured length distribution.
- **Tokenization.** Whitespace splitting left HTML-entity fragments and punctuation
  attached to words (`#39;s`, `quot;`, `(reuters)`, `said.`).
- **Inconsistent artifacts.** v1's RNN confusion matrix came from a different run than its
  reported metrics (63.8% vs 67.8% accuracy), and the data behind its sequence-length plot
  was missing. Every number is now generated from a single results file.
- **Cost measurement.** v1 reported the RNN as slower than the LSTM. That turned out to be a
  property of PyTorch's MPS backend (Section 5.6), not of the models. v1 also compared parameter
  counts that were 99% embedding table.
- **Baselines.** v1 had no reference outside the four neural models. This edition adds
  TF-IDF with logistic regression, and GloVe initialization for every neural model.

# AI assistance

Version 1 of this project was developed with the help of ChatGPT and OpenAI Codex. This
revised edition was developed with the help of Claude Code, which was used to review the
original code and results, restructure and test the codebase, run the experiments and draft
the text. I reviewed the code, checked the reported numbers against the saved results and
remain responsible for the content.

# References

::: {#refs}
:::

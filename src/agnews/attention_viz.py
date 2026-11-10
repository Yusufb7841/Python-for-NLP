"""Visualize where the Attention LSTM looks. Loads the seed-0 checkpoint saved by the main
suite, shades each token of a few test articles by its attention weight, and reports how much
attention mass falls on the opening tokens (roughly the headline).

    python -m agnews.attention_viz

Caveat: each attention weight is over an LSTM *state*, which already summarizes the words to
its left, so a weight is not a pure per-word importance score (Jain & Wallace, 2019).
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

from agnews import config  # noqa: E402
from agnews.data import LABEL_NAMES, PAD_ID, Vocabulary, encode_batch, load_corpus  # noqa: E402
from agnews.figures import DARK, LIGHT, caption, save, themed  # noqa: E402
from agnews.models import build_model  # noqa: E402

HEADLINE_TOKENS = 8
MIN_TOKENS, MAX_TOKENS = 18, 34  # keep examples short enough to read in a figure


def load_attention_model(embeddings="glove", seed=0):
    path = config.ARTIFACTS_DIR / f"attention-{embeddings}-seed{seed}.pt"
    checkpoint = torch.load(path, map_location="cpu")
    vocab = Vocabulary(checkpoint["vocab"])
    model = build_model(
        "attention", len(vocab), config.EMBEDDING_DIM, config.HIDDEN_DIM, config.DROPOUT
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), vocab, checkpoint["spec"]["max_length"]


@torch.no_grad()
def attend(model, input_ids, batch_size=512):
    probabilities, weights = [], []
    for start in range(0, len(input_ids), batch_size):
        logits, attention = model(input_ids[start : start + batch_size], return_attention=True)
        probabilities.append(logits.softmax(dim=1))
        weights.append(attention)
    return torch.cat(probabilities).numpy(), torch.cat(weights).numpy()


def pick_examples(tokens, labels, predictions):
    """One confident correct example per topic, then the first Business/Sci-Tech mix-ups."""
    lengths = np.array([len(t) for t in tokens])
    readable = (lengths >= MIN_TOKENS) & (lengths <= MAX_TOKENS)
    chosen = []
    for label in range(len(LABEL_NAMES)):
        candidates = np.flatnonzero(readable & (labels == label) & (predictions == label))
        chosen.append(int(candidates[0]))
    for true, predicted in [(2, 3), (3, 2)]:
        candidates = np.flatnonzero(readable & (labels == true) & (predictions == predicted))
        if len(candidates):
            chosen.append(int(candidates[0]))
    return chosen


def draw_examples(examples, theme):
    """Each example: a header line, then its tokens as chips shaded by attention weight
    (relative to that article's peak), wrapped to the figure width."""
    height = 0.33 * sum(e["lines"] + 2 for e in examples)
    fig = plt.figure(figsize=(10, height))
    renderer = fig.canvas.get_renderer()
    to_figure = fig.transFigure.inverted()
    line, margin, gap = 0.3 / height, 0.012, 0.009
    highlight = theme.series["fnn"]
    y = 1.0

    for example in examples:
        correct = example["true"] == example["predicted"]
        header = (
            f"True: {LABEL_NAMES[example['true']]}   ·   Predicted: "
            f"{LABEL_NAMES[example['predicted']]} ({example['confidence']:.0%})"
            + ("" if correct else "   ·   misclassified")
        )
        fig.text(margin, y, header, fontsize=9, fontweight="bold", va="top", color=theme.ink)
        y -= line * 1.1
        x = margin
        peak = example["weights"].max()
        for token, weight in zip(example["tokens"], example["weights"], strict=True):
            chip = fig.text(
                x, y, token, fontsize=9.5, va="top", color=theme.ink,
                bbox={"boxstyle": "round,pad=0.25,rounding_size=0.2", "linewidth": 0,
                      "facecolor": to_rgba(highlight, 0.85 * weight / peak)},
            )  # fmt: skip
            box = chip.get_window_extent(renderer).transformed(to_figure)
            if box.x1 > 1 - margin:  # wrap to the next line
                y -= line
                x = margin
                chip.set_position((x, y))
                box = chip.get_window_extent(renderer).transformed(to_figure)
            x = box.x1 + gap
        y -= line * 1.7
    caption(
        fig,
        "Shading = attention weight relative to the article's largest weight. Tokens are shown "
        "as the model sees them (<unk> = outside the vocabulary).",
        theme,
        gap=0.2,
    )
    return fig


def attention_stats(weights, lengths):
    """Average attention mass on the first and last HEADLINE_TOKENS positions, and what
    uniform attention over each article would give."""
    first = weights[:, :HEADLINE_TOKENS].sum(axis=1)
    last = np.array(
        [w[max(n - HEADLINE_TOKENS, 0) : n].sum() for w, n in zip(weights, lengths, strict=True)]
    )
    uniform = np.minimum(HEADLINE_TOKENS, lengths) / lengths
    return first.mean(), last.mean(), uniform.mean()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--embeddings", choices=["glove", "random"], default="glove",
                        help="which checkpoint to draw the examples from")  # fmt: skip
    args = parser.parse_args(argv)
    corpus = load_corpus()

    lines, drawn = [], None
    for embeddings in ("glove", "random"):
        if not (config.ARTIFACTS_DIR / f"attention-{embeddings}-seed0.pt").exists():
            continue
        model, vocab, max_length = load_attention_model(embeddings)
        input_ids = encode_batch(corpus.test_tokens, vocab, max_length)
        probabilities, weights = attend(model, input_ids)
        lengths = (input_ids != PAD_ID).sum(dim=1).numpy()
        accuracy = (probabilities.argmax(axis=1) == corpus.test_labels).mean()
        first, last, uniform = attention_stats(weights, lengths)
        lines.append(
            f"{embeddings} embeddings, seed 0 (test accuracy {accuracy:.2%}): attention mass on "
            f"the first {HEADLINE_TOKENS} tokens {first:.1%}, on the last {HEADLINE_TOKENS} "
            f"tokens {last:.1%}; uniform attention would give {uniform:.1%} to each."
        )
        print(lines[-1])
        if embeddings == args.embeddings:
            drawn = (vocab, input_ids, probabilities, weights, lengths)
    (config.RESULTS_DIR / "attention_stats.txt").write_text("\n".join(lines) + "\n")

    vocab, input_ids, probabilities, weights, lengths = drawn
    predictions = probabilities.argmax(axis=1)
    examples = []
    for index in pick_examples(corpus.test_tokens, corpus.test_labels, predictions):
        n = lengths[index]
        tokens = [vocab.itos[i] for i in input_ids[index, :n].tolist()]
        examples.append(
            {
                "tokens": tokens,
                "weights": weights[index, :n],
                "true": int(corpus.test_labels[index]),
                "predicted": int(predictions[index]),
                "confidence": float(probabilities[index].max()),
                "lines": 1 + len(" ".join(tokens)) // 110,
            }
        )
    for theme in (LIGHT, DARK):
        with themed(theme):
            print("Saved", save(draw_examples(examples, theme), "attention_examples", theme,
                                config.FIGURES_DIR))  # fmt: skip


if __name__ == "__main__":
    main()

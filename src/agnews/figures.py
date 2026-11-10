"""Regenerate every figure from results/*.csv. Each figure is written twice, NAME.png (light)
and NAME-dark.png, so the README can follow GitHub's light/dark theme.

    python -m agnews.figures
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from agnews import config  # noqa: E402
from agnews.metrics import parse_confusion  # noqa: E402
from agnews.summary import (  # noqa: E402
    CLASS_LABELS,
    MODEL_LABELS,
    MODEL_ORDER,
    estimated_cpu_minutes,
    learning_curve_runs,
    load_runs,
    load_timing,
    main_runs,
    sequence_length_runs,
)

NEURAL = ["fnn", "rnn", "lstm", "attention"]
MARKERS = {"tfidf": "s", "fnn": "o", "rnn": "v", "lstm": "D", "attention": "^"}
EMBEDDING_PANELS = [
    ("random", "Randomly initialized embeddings"),
    ("glove", "GloVe-initialized embeddings"),
]


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    series: dict
    ramp: list


# Categorical slots validated for adjacent-pair colour-vision separation in both modes;
# TF-IDF is the non-neural reference and is drawn in neutral ink rather than a hue.
LIGHT = Theme(
    "light", "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
    {"fnn": "#2a78d6", "rnn": "#eb6834", "lstm": "#1baf7a", "attention": "#eda100",
     "tfidf": "#52514e"},
    ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
)  # fmt: skip
DARK = Theme(
    "dark", "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835",
    {"fnn": "#3987e5", "rnn": "#d95926", "lstm": "#199e70", "attention": "#c98500",
     "tfidf": "#c3c2b7"},
    ["#1a1a19", "#16304f", "#184f95", "#2a78d6", "#6da7ec", "#cde2fb"],
)  # fmt: skip


@contextmanager
def themed(theme):
    params = {
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "axes.edgecolor": theme.axis,
        "axes.labelcolor": theme.ink2,
        "axes.titlecolor": theme.ink,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": theme.grid,
        "grid.linewidth": 0.8,
        "xtick.color": theme.axis,
        "ytick.color": theme.axis,
        "xtick.labelcolor": theme.ink2,
        "ytick.labelcolor": theme.ink2,
        "text.color": theme.ink,
        "legend.frameon": False,
        "legend.labelcolor": theme.ink2,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }
    with plt.rc_context(params):
        yield


def save(fig, name, theme, output_dir):
    suffix = "" if theme.name == "light" else "-dark"
    path = output_dir / f"{name}{suffix}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return path


def caption(fig, text, theme, gap=0.08):
    """Footnote placed `gap` inches below the lowest artist already drawn on the figure."""
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    x, y = bbox.x0 / fig.get_figwidth(), (bbox.y0 - gap) / fig.get_figheight()
    fig.text(x, y, text, ha="left", va="top", fontsize=8.5, color=theme.muted)


def percent_axis(axis):
    axis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=None))


def model_legend(theme, models, include_tfidf=True):
    handles = []
    for model in (["tfidf"] if include_tfidf else []) + models:
        handles.append(
            Line2D([], [], color=theme.series[model], marker=MARKERS[model], markersize=6,
                   markeredgecolor=theme.surface, label=MODEL_LABELS[model])
        )  # fmt: skip
    return handles


def plot_main_results(runs, timing, theme):
    """Two aligned dot plots: accuracy (left) and estimated CPU training time (right)."""
    main = main_runs(runs)
    stats = (
        main.groupby(["model", "embeddings"], observed=True)["test_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    minutes = estimated_cpu_minutes(main, timing)
    panels = 2 if minutes else 1
    fig, axes = plt.subplots(
        1, panels, figsize=(9.5 if panels == 2 else 6, 3.6), sharey=True,
        gridspec_kw={"width_ratios": [1.6, 1][:panels], "wspace": 0.08},
    )  # fmt: skip
    axes = np.atleast_1d(axes)
    rows = {model: i for i, model in enumerate(reversed(MODEL_ORDER))}
    offsets = {"random": -0.16, "glove": 0.16, "none": 0.0}

    def draw(ax, model, embeddings, x, xerr=None):
        y = rows[model] + offsets[embeddings]
        color = theme.series[model]
        if xerr is not None:
            ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor=color, elinewidth=1.4, capsize=0)
        hollow = embeddings == "random"
        ax.plot(
            x, y, marker=MARKERS[model], markersize=8, linestyle="none",
            markerfacecolor=theme.surface if hollow else color,
            markeredgecolor=color if hollow else theme.surface,
            markeredgewidth=1.6 if hollow else 1.0,
        )  # fmt: skip

    accuracy_ax = axes[0]
    tfidf_mean = stats.loc[stats["model"] == "tfidf", "mean"]
    if len(tfidf_mean):
        accuracy_ax.axvline(tfidf_mean.iloc[0], color=theme.muted, linewidth=0.8, zorder=0)
    for row in stats.itertuples(index=False):
        draw(accuracy_ax, row.model, row.embeddings, row.mean, row.std)
    for model, group in stats.groupby("model", observed=True):
        best = group.loc[group["mean"].idxmax()]
        accuracy_ax.annotate(
            f"{100 * best['mean']:.1f}%",
            (best["mean"] + (best["std"] if best["std"] == best["std"] else 0), rows[model]
             + offsets[best["embeddings"]]),
            xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5,
            color=theme.ink2,
        )  # fmt: skip
    low = stats["mean"].min() - stats["std"].fillna(0).max()
    high = stats["mean"].max()
    accuracy_ax.set_xlim(low - 0.01, high + 0.012)
    percent_axis(accuracy_ax.xaxis)
    accuracy_ax.set_xlabel("Test accuracy (7,600 articles)")
    accuracy_ax.set_yticks(list(rows.values()), [MODEL_LABELS[m] for m in rows])
    accuracy_ax.tick_params(axis="y", length=0)
    accuracy_ax.grid(axis="y", visible=False)
    accuracy_ax.spines["left"].set_visible(False)
    accuracy_ax.set_title("Accuracy")

    if panels == 2:
        time_ax = axes[1]
        for (model, embeddings), value in minutes.items():
            draw(time_ax, model, embeddings, value)
        time_ax.set_xscale("log")
        lo, hi = min(minutes.values()), max(minutes.values())
        ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50, 100, 200) if lo / 2 <= t <= hi * 2]
        time_ax.set_xticks(ticks, [f"{t:g} min" for t in ticks])
        time_ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        time_ax.set_xlim(lo / 1.6, hi * 1.6)
        time_ax.set_xlabel("Estimated training time on CPU (log scale)")
        time_ax.tick_params(axis="y", length=0)
        time_ax.grid(axis="y", visible=False)
        time_ax.spines["left"].set_visible(False)
        time_ax.set_title("Cost")

    key = [
        Line2D([], [], marker="o", linestyle="none", markersize=7, markerfacecolor=theme.surface,
               markeredgecolor=theme.ink2, markeredgewidth=1.6, label="random init embeddings"),
        Line2D([], [], marker="o", linestyle="none", markersize=7, markerfacecolor=theme.ink2,
               markeredgecolor=theme.surface, label="GloVe init embeddings"),
        Line2D([], [], color=theme.muted, linewidth=0.8, label="TF-IDF baseline"),
    ]  # fmt: skip
    fig.legend(handles=key, loc="lower left", bbox_to_anchor=(0.0, 0.98), ncol=3, fontsize=9)
    seeds = main.groupby(["model", "embeddings"], observed=True).size().min()
    caption(
        fig,
        f"Full training set, mean ± std over {seeds} seeds. Cost = controlled CPU seconds per "
        "epoch × epochs actually run (incl. early-stopping patience).",
        theme,
    )
    return fig


def plot_learning_curve(runs, theme):
    data = learning_curve_runs(runs)
    stats = (
        data.groupby(["model", "embeddings", "train_size"], observed=True)["test_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    sizes = sorted(stats["train_size"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (embeddings, title) in zip(axes, EMBEDDING_PANELS, strict=True):
        for model in MODEL_ORDER:
            subset = stats[
                (stats["model"] == model)
                & (stats["embeddings"] == ("none" if model == "tfidf" else embeddings))
            ].sort_values("train_size")
            if subset.empty:
                continue
            color = theme.series[model]
            ax.fill_between(
                subset["train_size"], subset["mean"] - subset["std"].fillna(0),
                subset["mean"] + subset["std"].fillna(0), color=color, alpha=0.12, linewidth=0,
            )  # fmt: skip
            ax.plot(
                subset["train_size"], subset["mean"], color=color, marker=MARKERS[model],
                markersize=5.5, markeredgecolor=theme.surface, markeredgewidth=1,
                linewidth=1.2 if model == "tfidf" else 2, zorder=3,
            )  # fmt: skip
        ax.set_xscale("log")
        ax.set_xticks(sizes, [f"{s // 1000}k" for s in sizes])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("Training examples (log scale)")
        ax.set_title(title)
        percent_axis(ax.yaxis)
    axes[0].set_ylabel("Test accuracy")
    fig.legend(
        handles=model_legend(theme, NEURAL), loc="lower left", bbox_to_anchor=(0.0, 0.98),
        ncol=5, fontsize=9,
    )  # fmt: skip
    caption(fig, "Mean over seeds; shaded band = ± 1 std. TF-IDF shown in both panels.", theme)
    return fig


def plot_sequence_length(runs, theme, truncated_share=None):
    stats = (
        sequence_length_runs(runs)
        .groupby(["model", "max_length"], observed=True)["test_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    lengths = sorted(stats["max_length"].unique())
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for model in MODEL_ORDER:
        subset = stats[stats["model"] == model].sort_values("max_length")
        color = theme.series[model]
        ax.fill_between(
            subset["max_length"], subset["mean"] - subset["std"].fillna(0),
            subset["mean"] + subset["std"].fillna(0), color=color, alpha=0.12, linewidth=0,
        )  # fmt: skip
        ax.plot(
            subset["max_length"], subset["mean"], color=color, marker=MARKERS[model],
            markersize=5.5, markeredgecolor=theme.surface, markeredgewidth=1,
            linewidth=1.2 if model == "tfidf" else 2, zorder=3,
        )  # fmt: skip
    ax.set_xscale("log", base=2)
    labels = [str(length) for length in lengths]
    if truncated_share:
        labels = [f"{length}\n{truncated_share[length]:.1%} cut" for length in lengths]
    ax.set_xticks(lengths, labels)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Tokens kept per article (and share of articles truncated)")
    ax.set_ylabel("Test accuracy")
    percent_axis(ax.yaxis)
    ax.legend(handles=model_legend(theme, NEURAL), loc="lower right", fontsize=8.5)
    ax.set_title("How much of each article do the models need?")
    caption(fig, "Full training set, random-init embeddings, mean ± std over seeds.", theme)
    return fig


def plot_confusion_matrices(runs, theme):
    main = main_runs(runs)
    best = main.groupby(["model", "embeddings"], observed=True)["test_accuracy"].mean()
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(13, 3.2), gridspec_kw={"wspace": 0.12})
    cmap = LinearSegmentedColormap.from_list("ramp", theme.ramp)
    for ax, model in zip(axes, MODEL_ORDER, strict=True):
        embeddings = best[model].idxmax()
        subset = main[(main["model"] == model) & (main["embeddings"] == embeddings)]
        matrix = sum(parse_confusion(value) for value in subset["confusion"])
        rates = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(rates, cmap=cmap, vmin=0, vmax=1)
        for (i, j), rate in np.ndenumerate(rates):
            rgb = to_rgb(cmap(rate))
            luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            ax.text(j, i, f"{100 * rate:.1f}", ha="center", va="center", fontsize=8,
                    color="#0b0b0b" if luminance > 0.5 else "#ffffff")  # fmt: skip
        ax.set_xticks(range(4), CLASS_LABELS, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(4), CLASS_LABELS if model == MODEL_ORDER[0] else [], fontsize=8)
        ax.tick_params(length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        suffix = {"none": "", "glove": " · GloVe", "random": " · random init"}[embeddings]
        ax.set_title(f"{MODEL_LABELS[model]}{suffix}", fontsize=9)
    axes[0].set_ylabel("True topic")
    axes[len(axes) // 2].set_xlabel("Predicted topic", labelpad=8)
    caption(
        fig,
        "Row-normalized % of each true topic, summed over seeds; best embedding setting per model.",
        theme,
    )
    return fig


def plot_training_curves(history, runs, theme):
    main = main_runs(runs)
    merged = history.merge(main[["key", "model", "embeddings"]], on="key")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (embeddings, title) in zip(axes, EMBEDDING_PANELS, strict=True):
        for model in NEURAL:
            subset = merged[(merged["model"] == model) & (merged["embeddings"] == embeddings)]
            curve = subset.groupby("epoch")["val_accuracy"].mean()
            ax.plot(curve.index, curve.values, color=theme.series[model], marker=MARKERS[model],
                    markersize=4.5, markeredgecolor=theme.surface, markeredgewidth=0.8)  # fmt: skip
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_title(title)
        percent_axis(ax.yaxis)
    axes[0].set_ylabel("Validation accuracy")
    fig.legend(
        handles=model_legend(theme, NEURAL, include_tfidf=False), loc="lower left",
        bbox_to_anchor=(0.0, 0.98), ncol=4, fontsize=9,
    )  # fmt: skip
    caption(
        fig, "Mean over seeds of the runs still training at each epoch (early stopping).", theme
    )
    return fig


def truncation_shares(lengths):
    from agnews.data import load_corpus

    token_counts = np.array([len(tokens) for tokens in load_corpus().train_tokens])
    return {length: float((token_counts > length).mean()) for length in lengths}


def main(argv=None):
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", default=config.FIGURES_DIR)
    args = parser.parse_args(argv)
    output_dir = config.PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs()
    timing = load_timing()
    history = pd.read_csv(config.RESULTS_DIR / "history.csv")
    lengths = sorted(sequence_length_runs(runs)["max_length"].unique())
    shares = truncation_shares(lengths)

    for theme in (LIGHT, DARK):
        with themed(theme):
            for name, fig in [
                ("main_results", plot_main_results(runs, timing, theme)),
                ("learning_curve", plot_learning_curve(runs, theme)),
                ("sequence_length", plot_sequence_length(runs, theme, shares)),
                ("confusion_matrices", plot_confusion_matrices(runs, theme)),
                ("training_curves", plot_training_curves(history, runs, theme)),
            ]:
                print("Saved", save(fig, name, theme, output_dir))


if __name__ == "__main__":
    main()

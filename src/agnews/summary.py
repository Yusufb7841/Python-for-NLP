"""Aggregate results/runs.csv into mean ± std tables. Writes results/summary.md and refreshes
the tables embedded in README.md between `<!-- BEGIN table:NAME -->` / `<!-- END ... -->`
markers, so no number in the write-up is typed by hand.

    python -m agnews.summary
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from agnews import config
from agnews.metrics import CLASS_KEYS

MODEL_ORDER = ["tfidf", "fnn", "rnn", "lstm", "attention"]
MODEL_LABELS = {
    "tfidf": "TF-IDF + LogReg",
    "fnn": "FNN (mean-pooled)",
    "rnn": "RNN",
    "lstm": "LSTM",
    "attention": "Attention LSTM",
}
EMBEDDING_LABELS = {"none": "n/a", "random": "random", "glove": "GloVe"}
CLASS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def load_runs(path=config.RESULTS_DIR / "runs.csv"):
    # Two workers sharing a queue can both finish the same run; count it once.
    runs = pd.read_csv(path).drop_duplicates("key", keep="first")
    runs["full_data"] = runs["key"].str.split("|").str[2] == "full"
    runs["model"] = pd.Categorical(runs["model"], MODEL_ORDER, ordered=True)
    return runs


def load_timing(path=config.RESULTS_DIR / "timing.csv"):
    return pd.read_csv(path) if path.exists() else None


def main_runs(runs):
    return runs[runs["full_data"] & (runs["max_length"] == config.MAX_LENGTH)]


def learning_curve_runs(runs):
    return runs[runs["max_length"] == config.MAX_LENGTH]


def sequence_length_runs(runs):
    return runs[runs["full_data"] & runs["embeddings"].isin(["random", "none"])]


def aggregate(runs, by, columns=("test_accuracy", "test_macro_f1")):
    grouped = runs.groupby(list(by), observed=True)
    table = grouped[list(columns)].agg(["mean", "std"])
    table.columns = [f"{column}_{stat}" for column, stat in table.columns]
    table["seeds"] = grouped.size()
    return table.reset_index()


def estimated_cpu_minutes(main, timing):
    """Controlled CPU seconds/epoch x the number of epochs each config actually ran."""
    if timing is None:
        return {}
    cpu = timing[timing["device"] == "cpu"].set_index("model")["seconds"]
    minutes = {}
    for (model, embeddings), group in main.groupby(["model", "embeddings"], observed=True):
        if model == "tfidf":
            minutes[(model, embeddings)] = cpu["tfidf"] / 60
        else:
            minutes[(model, embeddings)] = cpu[model] * group["epochs_run"].mean() / 60
    return minutes


def pct(mean, std=None, digits=1):
    if std is None or pd.isna(std):
        return f"{100 * mean:.{digits}f}"
    return f"{100 * mean:.{digits}f} ± {100 * std:.{digits}f}"


NUMERIC_CELL = re.compile(r"^(?:[\d.,± ]+|n/a|)$")


def markdown(frame):
    """Pipe table; text columns left-aligned, numeric columns right-aligned."""
    numeric = [all(NUMERIC_CELL.match(str(v)) for v in frame[c]) for c in frame.columns]
    header = "| " + " | ".join(frame.columns) + " |"
    align = "| " + " | ".join("--:" if n else ":--" for n in numeric) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, align, *rows])


def main_table(runs, timing):
    main = main_runs(runs)
    stats = aggregate(main, ["model", "embeddings"])
    epochs = main.groupby(["model", "embeddings"], observed=True)["best_epoch"].mean()
    params = main.groupby(["model", "embeddings"], observed=True)[
        "non_embedding_parameters"
    ].first()
    minutes = estimated_cpu_minutes(main, timing)
    rows = []
    for row in stats.itertuples(index=False):
        key = (row.model, row.embeddings)
        rows.append(
            {
                "Model": MODEL_LABELS[row.model],
                "Embeddings": EMBEDDING_LABELS[row.embeddings],
                "Test accuracy (%)": pct(row.test_accuracy_mean, row.test_accuracy_std, 2),
                "Macro F1 (%)": pct(row.test_macro_f1_mean, row.test_macro_f1_std, 2),
                "Best epoch": "n/a" if pd.isna(epochs[key]) else f"{epochs[key]:.1f}",
                "CPU train time (min)": f"{minutes[key]:.1f}" if key in minutes else "",
                "Params excl. embeddings": (
                    "n/a" if pd.isna(params[key]) else f"{int(params[key]):,}"
                ),
            }
        )
    return markdown(pd.DataFrame(rows))


def per_class_table(runs):
    main = main_runs(runs)
    columns = [f"f1_{key}" for key in CLASS_KEYS]
    means = main.groupby(["model", "embeddings"], observed=True)[columns].mean().reset_index()
    rows = []
    for row in means.itertuples(index=False):
        values = {
            label: pct(getattr(row, col)) for label, col in zip(CLASS_LABELS, columns, strict=True)
        }
        rows.append(
            {
                "Model": MODEL_LABELS[row.model],
                "Embeddings": EMBEDDING_LABELS[row.embeddings],
                **values,
            }
        )
    return markdown(pd.DataFrame(rows))


def learning_curve_table(runs):
    stats = aggregate(learning_curve_runs(runs), ["model", "embeddings", "train_size"])
    sizes = sorted(stats["train_size"].unique())
    rows = []
    for (model, embeddings), group in stats.groupby(["model", "embeddings"], observed=True):
        by_size = group.set_index("train_size")
        row = {"Model": MODEL_LABELS[model], "Embeddings": EMBEDDING_LABELS[embeddings]}
        for size in sizes:
            cell = by_size.loc[size] if size in by_size.index else None
            label = f"{size:,}" if size < max(sizes) else f"{size:,} (all)"
            row[label] = "" if cell is None else pct(cell.test_accuracy_mean)
        rows.append(row)
    return markdown(pd.DataFrame(rows))


def sequence_length_table(runs):
    stats = aggregate(sequence_length_runs(runs), ["model", "max_length"])
    pivot = stats.pivot(index="model", columns="max_length", values="test_accuracy_mean")
    rows = []
    for model, values in pivot.iterrows():
        row = {"Model": MODEL_LABELS[model]}
        row.update({f"{length} tokens": pct(values[length]) for length in pivot.columns})
        rows.append(row)
    return markdown(pd.DataFrame(rows))


def timing_table(timing):
    if timing is None:
        return "_Run `python -m agnews.benchmark` to produce timing results._"
    neural = timing[timing["unit"] == "epoch"]
    pivot = neural.pivot(index="model", columns="device", values="seconds")
    rows = []
    for model in [m for m in MODEL_ORDER if m in pivot.index]:
        row = {"Model": MODEL_LABELS[model]}
        row.update(
            {f"{device.upper()} s/epoch": f"{pivot.loc[model, device]:.1f}" for device in pivot}
        )
        rows.append(row)
    tfidf = timing[timing["model"] == "tfidf"]
    note = ""
    if len(tfidf):
        seconds = tfidf["seconds"].iloc[0]
        note = f"\n\nTF-IDF + LogReg: {seconds:.0f} s for a full CPU fit (including the C search)."
    return markdown(pd.DataFrame(rows)) + note


def build_tables(runs, timing):
    return {
        "main": main_table(runs, timing),
        "per_class": per_class_table(runs),
        "learning_curve": learning_curve_table(runs),
        "sequence_length": sequence_length_table(runs),
        "timing": timing_table(timing),
    }


def inject_tables(text, tables):
    for name, table in tables.items():
        pattern = re.compile(
            rf"(<!-- BEGIN table:{name} -->).*?(<!-- END table:{name} -->)", re.DOTALL
        )
        text = pattern.sub(lambda m, t=table: f"{m.group(1)}\n{t}\n{m.group(2)}", text)
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--readme", type=Path, default=config.PROJECT_ROOT / "README.md")
    args = parser.parse_args(argv)

    tables = build_tables(load_runs(), load_timing())
    sections = [f"## {name.replace('_', ' ').title()}\n\n{table}" for name, table in tables.items()]
    summary_path = config.RESULTS_DIR / "summary.md"
    summary_path.write_text(
        "# Results summary\n\nGenerated by `python -m agnews.summary` from `results/runs.csv`. "
        "Accuracy is on the full 7,600-example AG News test set; ± is the standard deviation "
        "across seeds.\n\n" + "\n\n".join(sections) + "\n"
    )
    print(f"Wrote {summary_path}")
    if args.readme.exists():
        args.readme.write_text(inject_tables(args.readme.read_text(), tables))
        print(f"Updated tables in {args.readme}")


if __name__ == "__main__":
    main()

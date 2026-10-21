import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", default="results/main_metrics.csv")
    parser.add_argument("--output-dir", default="results/figures")
    return parser.parse_args()


def load_metrics(metrics_path):
    return pd.read_csv(metrics_path)


def get_final_rows(metrics):
    final_indices = metrics.groupby("model")["epoch"].idxmax()
    return metrics.loc[final_indices].sort_values("model")


def plot_validation_accuracy(metrics, output_dir):
    plt.figure(figsize=(8, 5))
    for model_name, model_metrics in metrics.groupby("model"):
        plt.plot(model_metrics["epoch"], model_metrics["validation_accuracy"], marker="o", label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()
    plt.tight_layout()
    output_path = output_dir / "validation_accuracy.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_validation_loss(metrics, output_dir):
    plt.figure(figsize=(8, 5))
    for model_name, model_metrics in metrics.groupby("model"):
        plt.plot(model_metrics["epoch"], model_metrics["validation_loss"], marker="o", label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Validation Loss Over Epochs")
    plt.legend()
    plt.tight_layout()
    output_path = output_dir / "validation_loss.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_final_test_accuracy(final_rows, output_dir):
    plt.figure(figsize=(8, 5))
    plt.bar(final_rows["model"], final_rows["test_accuracy"])
    plt.xlabel("Model")
    plt.ylabel("Test accuracy")
    plt.title("Final Test Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    output_path = output_dir / "final_test_accuracy.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_training_time(final_rows, output_dir):
    plt.figure(figsize=(8, 5))
    plt.bar(final_rows["model"], final_rows["training_time_seconds"])
    plt.xlabel("Model")
    plt.ylabel("Training time seconds")
    plt.title("Training Time Comparison")
    plt.tight_layout()
    output_path = output_dir / "training_time.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def print_summary(final_rows):
    summary = final_rows[
        [
            "model",
            "validation_accuracy",
            "test_accuracy",
            "parameter_count",
            "training_time_seconds",
        ]
    ].copy()
    summary["validation_accuracy"] = summary["validation_accuracy"].map(lambda value: f"{value:.4f}")
    summary["test_accuracy"] = summary["test_accuracy"].map(lambda value: f"{value:.4f}")
    summary["training_time_seconds"] = summary["training_time_seconds"].map(lambda value: f"{value:.2f}")
    print(summary.to_string(index=False))


def main():
    args = parse_args()
    metrics_path = Path(args.metrics_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(metrics_path)
    final_rows = get_final_rows(metrics)

    figure_paths = [
        plot_validation_accuracy(metrics, output_dir),
        plot_validation_loss(metrics, output_dir),
        plot_final_test_accuracy(final_rows, output_dir),
        plot_training_time(final_rows, output_dir),
    ]

    print("Summary table")
    print_summary(final_rows)
    print("Generated figures")
    for figure_path in figure_paths:
        print(figure_path)


if __name__ == "__main__":
    main()

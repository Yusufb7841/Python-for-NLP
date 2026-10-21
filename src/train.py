import argparse
import csv
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from data import LABEL_NAMES, create_data_loaders
from models import get_model


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def calculate_accuracy(logits, labels):
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return correct


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run_one_epoch(model, data_loader, loss_function, optimizer, device, training):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(input_ids)
            loss = loss_function(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += calculate_accuracy(logits, labels)
        total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def evaluate_with_predictions(model, data_loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    rows = []
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids)
            loss = loss_function(logits, labels)
            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_examples += batch_size

            labels_list = labels.cpu().tolist()
            predictions_list = predictions.cpu().tolist()
            true_labels.extend(labels_list)
            predicted_labels.extend(predictions_list)

            for row_index, true_label, predicted_label, text in zip(
                batch["example_index"],
                labels_list,
                predictions_list,
                batch["text"],
            ):
                rows.append(
                    {
                        "example_index": int(row_index),
                        "text": text,
                        "true_label_id": true_label,
                        "true_label_name": LABEL_NAMES[true_label],
                        "predicted_label_id": predicted_label,
                        "predicted_label_name": LABEL_NAMES[predicted_label],
                        "correct": true_label == predicted_label,
                    }
                )

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=list(range(len(LABEL_NAMES))),
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(LABEL_NAMES))),
    )

    class_metrics = []
    for label_id, label_name in enumerate(LABEL_NAMES):
        class_metrics.append(
            {
                "label_id": label_id,
                "label_name": label_name,
                "precision": precision[label_id],
                "recall": recall[label_id],
                "f1": f1[label_id],
                "support": support[label_id],
            }
        )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "class_metrics": class_metrics,
        "confusion_matrix": matrix,
        "predictions": rows,
    }


def save_confusion_matrix_plot(matrix, output_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(len(LABEL_NAMES)), LABEL_NAMES, rotation=30, ha="right")
    plt.yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
    plt.colorbar()

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            plt.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_error_examples(predictions, output_path, model_name, num_examples):
    incorrect_rows = [row for row in predictions if not row["correct"]]

    with output_path.open("w") as file:
        file.write("# Error Examples\n\n")
        file.write(f"Model: `{model_name}`\n\n")

        if not incorrect_rows:
            file.write("No misclassified examples found.\n")
            return

        for index, row in enumerate(incorrect_rows[:num_examples], start=1):
            text_excerpt = row["text"].replace("\n", " ")[:400]
            file.write(f"## Example {index}\n\n")
            file.write(f"- True label: {row['true_label_name']} ({row['true_label_id']})\n")
            file.write(f"- Predicted label: {row['predicted_label_name']} ({row['predicted_label_id']})\n")
            file.write(f"- Text excerpt: {text_excerpt}\n\n")


def save_analysis_files(analysis, analysis_dir, run_id, model_name, num_error_examples):
    analysis_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = f"{run_id}_{model_name}"

    predictions_path = analysis_dir / f"{file_prefix}_predictions.csv"
    class_metrics_path = analysis_dir / f"{file_prefix}_class_metrics.csv"
    confusion_csv_path = analysis_dir / f"{file_prefix}_confusion_matrix.csv"
    confusion_png_path = analysis_dir / f"{file_prefix}_confusion_matrix.png"
    error_examples_path = analysis_dir / f"{file_prefix}_error_examples.md"

    pd.DataFrame(analysis["predictions"]).to_csv(predictions_path, index=False)
    pd.DataFrame(analysis["class_metrics"]).to_csv(class_metrics_path, index=False)
    pd.DataFrame(
        analysis["confusion_matrix"],
        index=LABEL_NAMES,
        columns=LABEL_NAMES,
    ).to_csv(confusion_csv_path)
    save_confusion_matrix_plot(analysis["confusion_matrix"], confusion_png_path)
    save_error_examples(
        analysis["predictions"],
        error_examples_path,
        model_name,
        num_error_examples,
    )

    return {
        "predictions": predictions_path,
        "class_metrics": class_metrics_path,
        "confusion_csv": confusion_csv_path,
        "confusion_png": confusion_png_path,
        "error_examples": error_examples_path,
    }


def save_metrics(metrics, output_path, append_metrics):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    mode = "a" if append_metrics else "w"
    fieldnames = [
        "run_id",
        "model",
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "test_loss",
        "test_accuracy",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "parameter_count",
        "training_time_seconds",
    ]

    with output_path.open(mode, newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        if not append_metrics or not file_exists:
            writer.writeheader()
        writer.writerows(metrics)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fnn", "rnn", "lstm", "attention"], default="fnn")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=30)
    parser.add_argument("--max-vocab-size", type=int, default=3000)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--validation-limit", type=int, default=40)
    parser.add_argument("--test-limit", type=int, default=40)
    parser.add_argument("--output-path", default="results/metrics.csv")
    parser.add_argument("--append-metrics", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--save-analysis", action="store_true")
    parser.add_argument("--analysis-dir", default="results/analysis")
    parser.add_argument("--num-error-examples", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_loader, validation_loader, test_loader, vocabulary = create_data_loaders(
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_vocab_size=args.max_vocab_size,
        min_frequency=1,
        seed=args.seed,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
    )

    model = get_model(
        args.model,
        vocab_size=len(vocabulary),
        num_classes=4,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    metrics = []
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    parameter_count = count_parameters(model)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else Path("results") / "checkpoints" / f"{run_id}.pt"
    start_time = time.perf_counter()
    final_validation_accuracy = 0.0
    final_test_accuracy = 0.0
    final_test_macro_precision = 0.0
    final_test_macro_recall = 0.0
    final_test_macro_f1 = 0.0

    print("Run settings")
    print("Model:", args.model)
    print("Device:", device)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Max length:", args.max_length)
    print("Max vocab size:", args.max_vocab_size)
    print("Train limit:", args.train_limit)
    print("Validation limit:", args.validation_limit)
    print("Test limit:", args.test_limit)
    print("Output path:", args.output_path)
    print("Append metrics:", args.append_metrics)
    print("Save checkpoint:", args.save_checkpoint)
    print("Save analysis:", args.save_analysis)
    if args.save_analysis:
        print("Analysis dir:", args.analysis_dir)
        print("Number of error examples:", args.num_error_examples)
    if args.save_checkpoint:
        print("Checkpoint path:", checkpoint_path)
    print("Vocabulary size:", len(vocabulary))
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))
    print("Test batches:", len(test_loader))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
            training=True,
        )
        validation_loss, validation_accuracy = run_one_epoch(
            model,
            validation_loader,
            loss_function,
            optimizer,
            device,
            training=False,
        )
        test_loss = ""
        test_accuracy = ""
        test_macro_precision = ""
        test_macro_recall = ""
        test_macro_f1 = ""
        training_time_seconds = ""
        final_analysis = None

        if epoch == args.epochs:
            final_analysis = evaluate_with_predictions(
                model,
                test_loader,
                loss_function,
                device,
            )
            test_loss = final_analysis["loss"]
            test_accuracy = final_analysis["accuracy"]
            test_macro_precision = final_analysis["macro_precision"]
            test_macro_recall = final_analysis["macro_recall"]
            test_macro_f1 = final_analysis["macro_f1"]
            training_time_seconds = time.perf_counter() - start_time
            final_validation_accuracy = validation_accuracy
            final_test_accuracy = test_accuracy
            final_test_macro_precision = test_macro_precision
            final_test_macro_recall = test_macro_recall
            final_test_macro_f1 = test_macro_f1

        row = {
            "run_id": run_id,
            "model": args.model,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_macro_precision": test_macro_precision,
            "test_macro_recall": test_macro_recall,
            "test_macro_f1": test_macro_f1,
            "parameter_count": parameter_count,
            "training_time_seconds": training_time_seconds,
        }
        metrics.append(row)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"train_accuracy={train_accuracy:.4f}, "
            f"validation_loss={validation_loss:.4f}, "
            f"validation_accuracy={validation_accuracy:.4f}"
        )

    metrics_path = Path(args.output_path)
    save_metrics(metrics, metrics_path, args.append_metrics)
    print("Saved metrics to:", metrics_path)

    analysis_paths = {}
    if args.save_analysis and final_analysis is not None:
        analysis_paths = save_analysis_files(
            final_analysis,
            Path(args.analysis_dir),
            run_id,
            args.model,
            args.num_error_examples,
        )
        print("Saved analysis files")
        for analysis_name, analysis_path in analysis_paths.items():
            print(f"{analysis_name}: {analysis_path}")

    saved_checkpoint_path = ""
    if args.save_checkpoint:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "run_id": run_id,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "vocab_size": len(vocabulary),
                "num_classes": 4,
                "embedding_dim": args.embedding_dim,
                "hidden_dim": args.hidden_dim,
                "max_length": args.max_length,
            },
            checkpoint_path,
        )
        saved_checkpoint_path = str(checkpoint_path)

    print("Final summary")
    print("Model:", args.model)
    print("Final validation accuracy:", f"{final_validation_accuracy:.4f}")
    print("Final test accuracy:", f"{final_test_accuracy:.4f}")
    print("Final test macro precision:", f"{final_test_macro_precision:.4f}")
    print("Final test macro recall:", f"{final_test_macro_recall:.4f}")
    print("Final test macro F1:", f"{final_test_macro_f1:.4f}")
    print("Parameter count:", parameter_count)
    print("Training time seconds:", f"{training_time_seconds:.2f}")
    print("Metrics path:", metrics_path)
    if saved_checkpoint_path:
        print("Checkpoint path:", saved_checkpoint_path)


if __name__ == "__main__":
    main()

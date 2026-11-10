"""Run the experiment suites. Each finished run appends one row to results/runs.csv (and its
per-epoch curve to results/history.csv); runs already in runs.csv are skipped, so an
interrupted sweep can simply be restarted.

    python -m agnews.experiments --suites main learning_curve sequence_length --seeds 0 1 2
"""

import argparse
import csv
import itertools
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from agnews import config
from agnews.baselines import TfidfLogReg
from agnews.data import Vocabulary, encode_batch, load_corpus, split_indices
from agnews.embeddings import glove_matrix
from agnews.metrics import classification_metrics
from agnews.models import NEURAL_MODELS, build_model, count_parameters
from agnews.training import predict, resolve_device, set_seed, train_model

try:
    import fcntl
except ImportError:  # Windows: no locking, so don't run workers in parallel there.
    fcntl = None

ALL_MODELS = ("tfidf",) + NEURAL_MODELS
FULL = None  # train_size meaning "the whole training split"

SUITES = {
    "main": {
        "models": ALL_MODELS,
        "embeddings": ("random", "glove"),
        "train_sizes": (FULL,),
        "max_lengths": (config.MAX_LENGTH,),
    },
    "learning_curve": {
        "models": ALL_MODELS,
        "embeddings": ("random", "glove"),
        "train_sizes": (1000, 3000, 10000, 30000),
        "max_lengths": (config.MAX_LENGTH,),
    },
    "sequence_length": {
        "models": ALL_MODELS,
        "embeddings": ("random",),
        "train_sizes": (FULL,),
        "max_lengths": (8, 16, 32),
    },
}


@dataclass(frozen=True)
class RunSpec:
    suite: str
    model: str
    embeddings: str  # "random" | "glove" | "none" (TF-IDF)
    train_size: int | None
    max_length: int
    seed: int

    @property
    def key(self):
        size = "full" if self.train_size is None else self.train_size
        return f"{self.model}|{self.embeddings}|{size}|{self.max_length}|{self.seed}"


def expand_suites(suite_names, seeds, models=None):
    specs, seen = [], set()
    for name in suite_names:
        suite = SUITES[name]
        for model, embeddings, size, length, seed in itertools.product(
            suite["models"], suite["embeddings"], suite["train_sizes"], suite["max_lengths"], seeds
        ):
            if models and model not in models:
                continue
            spec = RunSpec(
                name, model, "none" if model == "tfidf" else embeddings, size, length, seed
            )
            if spec.key not in seen:  # TF-IDF ignores the embeddings axis
                seen.add(spec.key)
                specs.append(spec)
    return specs


def completed_keys(runs_path):
    if not runs_path.exists():
        return set()
    return set(pd.read_csv(runs_path, usecols=["key"])["key"])


def append_rows(path, rows):
    """Append rows to a CSV, writing the header if the file is new. Locked so that
    several worker processes can share one results file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as file:
        if fcntl:
            fcntl.flock(file, fcntl.LOCK_EX)
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)
        file.flush()
        if fcntl:
            fcntl.flock(file, fcntl.LOCK_UN)


class DataCache:
    """Splits, vocabularies and encoded tensors are shared between runs in one process."""

    def __init__(self):
        self.corpus = load_corpus()
        self._splits, self._vocabs, self._encoded = {}, {}, {}

    def split(self, seed, train_size):
        key = (seed, train_size)
        if key not in self._splits:
            self._splits[key] = split_indices(
                self.corpus.train_labels, seed, config.VALIDATION_FRACTION, train_size
            )
        return self._splits[key]

    def tokens(self, seed, train_size):
        train_idx, val_idx = self.split(seed, train_size)
        tokens = self.corpus.train_tokens
        return [tokens[i] for i in train_idx], [tokens[i] for i in val_idx]

    def vocab(self, seed, train_size):
        key = (seed, train_size)
        if key not in self._vocabs:
            train_tokens, _ = self.tokens(seed, train_size)
            self._vocabs[key] = Vocabulary.build(
                train_tokens, config.MAX_VOCAB_SIZE, config.MIN_TOKEN_FREQ
            )
        return self._vocabs[key]

    def encoded(self, seed, train_size, max_length):
        key = (seed, train_size, max_length)
        if key not in self._encoded:
            vocab = self.vocab(seed, train_size)
            train_idx, val_idx = self.split(seed, train_size)
            labels = torch.as_tensor(self.corpus.train_labels)
            train_tokens, val_tokens = self.tokens(seed, train_size)
            self._encoded[key] = {
                "train": (encode_batch(train_tokens, vocab, max_length), labels[train_idx]),
                "val": (encode_batch(val_tokens, vocab, max_length), labels[val_idx]),
                "test": (
                    encode_batch(self.corpus.test_tokens, vocab, max_length),
                    torch.as_tensor(self.corpus.test_labels),
                ),
            }
        return self._encoded[key]


def run_tfidf(spec, data):
    train_tokens, val_tokens = data.tokens(spec.seed, spec.train_size)
    train_idx, val_idx = data.split(spec.seed, spec.train_size)
    labels = data.corpus.train_labels

    def truncate(token_lists):
        return [tokens[: spec.max_length] for tokens in token_lists]

    start = time.perf_counter()
    model = TfidfLogReg().fit(
        truncate(train_tokens), labels[train_idx], truncate(val_tokens), labels[val_idx]
    )
    train_seconds = time.perf_counter() - start
    predictions = model.predict(truncate(data.corpus.test_tokens))
    print(f"  best C={model.best_c}  val_acc {model.best_val_accuracy:.4f}  ({train_seconds:.1f}s)")
    row = {
        "train_size": len(train_idx),
        "device": "cpu",
        "best_epoch": "",
        "epochs_run": "",
        "train_seconds": train_seconds,
        "val_accuracy": model.best_val_accuracy,
        "parameters": model.num_features * 4 + 4,
        "non_embedding_parameters": "",
        "vocab_size": model.num_features,
        "glove_coverage": "",
    }
    return row, classification_metrics(data.corpus.test_labels, predictions), []


def run_neural(spec, data, device, checkpoint_dir=None):
    set_seed(spec.seed)
    encoded = data.encoded(spec.seed, spec.train_size, spec.max_length)
    vocab = data.vocab(spec.seed, spec.train_size)
    pretrained, coverage = None, ""
    if spec.embeddings == "glove":
        pretrained, coverage = glove_matrix(vocab, spec.seed)
        print(f"  GloVe coverage of vocabulary: {coverage:.1%}")
    model = build_model(
        spec.model,
        len(vocab),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT,
        pretrained=pretrained,
    ).to(device)

    result = train_model(model, encoded["train"], encoded["val"], device, spec.seed)
    _, predictions = predict(model, *encoded["test"], device)

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{spec.model}-{spec.embeddings}-seed{spec.seed}.pt"
        torch.save(
            {
                "spec": asdict(spec),
                "vocab": vocab.itos,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            },
            path,
        )
        print(f"  saved checkpoint {path}")

    row = {
        "train_size": len(encoded["train"][1]),
        "device": device.type,
        "best_epoch": result.best_epoch,
        "epochs_run": len(result.history),
        "train_seconds": result.train_seconds,
        "val_accuracy": result.best_val_accuracy,
        "parameters": count_parameters(model),
        "non_embedding_parameters": count_parameters(model, include_embedding=False),
        "vocab_size": len(vocab),
        "glove_coverage": coverage,
    }
    test_labels = encoded["test"][1].numpy()
    return row, classification_metrics(test_labels, predictions), result.history


def wants_checkpoint(spec):
    """Keep the weights we need for the attention visualizations, and nothing else."""
    return spec.suite == "main" and spec.model == "attention" and spec.seed == 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suites", nargs="+", choices=list(SUITES), default=list(SUITES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, help="restrict to these models")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="work through pending runs from the end, so a second worker can share a queue",
    )
    parser.add_argument("--dry-run", action="store_true", help="list pending runs and exit")
    args = parser.parse_args(argv)

    runs_path = args.results_dir / "runs.csv"
    history_path = args.results_dir / "history.csv"
    done = completed_keys(runs_path)
    pending = [s for s in expand_suites(args.suites, args.seeds, args.models) if s.key not in done]
    if args.reverse:
        pending.reverse()
    print(f"{len(pending)} runs pending ({len(done)} already in {runs_path})")
    if args.dry_run:
        for spec in pending:
            print(" ", spec.suite, spec.key)
        return

    device = resolve_device(args.device)
    data = DataCache()
    sweep_start = time.perf_counter()
    for number, spec in enumerate(pending, start=1):
        if spec.key in completed_keys(runs_path):  # finished by another worker meanwhile
            continue
        print(f"[{number}/{len(pending)}] {spec.suite}: {spec.key}  on {device.type}")
        if spec.model == "tfidf":
            row, metrics, history = run_tfidf(spec, data)
        else:
            checkpoint_dir = config.ARTIFACTS_DIR if wants_checkpoint(spec) else None
            row, metrics, history = run_neural(spec, data, device, checkpoint_dir)
        print(
            f"  -> test_acc {metrics['test_accuracy']:.4f}  macro_f1 {metrics['test_macro_f1']:.4f}"
        )
        identity = {"key": spec.key, **asdict(spec)}
        identity["train_size"] = row.pop("train_size")
        finished = {"finished_at": datetime.now().isoformat(timespec="seconds")}
        append_rows(runs_path, [{**identity, **row, **metrics, **finished}])
        if history:
            append_rows(history_path, [{"key": spec.key, **epoch} for epoch in history])
    print(f"Done in {(time.perf_counter() - sweep_start) / 60:.1f} min")


if __name__ == "__main__":
    main()

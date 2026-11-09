"""Controlled timing benchmark: training seconds per epoch for every model on every available
device, measured one at a time on an otherwise idle machine. Training runs in the sweep may
share the machine, so their own wall-clock times are not used for cost comparisons.

    python -m agnews.benchmark
"""

import argparse
import platform
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from agnews import config
from agnews.baselines import TfidfLogReg
from agnews.data import iterate_batches
from agnews.experiments import DataCache
from agnews.models import NEURAL_MODELS, build_model
from agnews.training import set_seed, synchronize

SAMPLE_SIZE = 20000  # examples timed per repeat; scaled up to a full epoch


def seconds_per_epoch(name, device, train, vocab_size, repeats=3):
    set_seed(0)
    model = build_model(name, vocab_size, config.EMBEDDING_DIM, config.HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    ids, labels = train[0][:SAMPLE_SIZE], train[1][:SAMPLE_SIZE]
    generator = torch.Generator().manual_seed(0)

    def one_pass(limit=None):
        model.train()
        for step, (x, y) in enumerate(
            iterate_batches(ids, labels, config.BATCH_SIZE, shuffle=True, generator=generator)
        ):
            if limit is not None and step >= limit:
                break
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
        synchronize(device)

    one_pass(limit=30)  # warm-up: kernel selection, allocator, caches
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        one_pass()
        timings.append(time.perf_counter() - start)
    return min(timings) * len(train[1]) / len(labels)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default=config.RESULTS_DIR / "timing.csv")
    args = parser.parse_args(argv)

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    devices += ["cuda"] if torch.cuda.is_available() else []
    data = DataCache()
    encoded = data.encoded(seed=0, train_size=None, max_length=config.MAX_LENGTH)
    vocab_size = len(data.vocab(seed=0, train_size=None))
    rows = []

    train_tokens, val_tokens = (
        [tokens[: config.MAX_LENGTH] for tokens in split]
        for split in data.tokens(seed=0, train_size=None)
    )
    train_idx, val_idx = data.split(seed=0, train_size=None)
    labels = data.corpus.train_labels
    start = time.perf_counter()
    TfidfLogReg().fit(train_tokens, labels[train_idx], val_tokens, labels[val_idx])
    rows.append(
        {
            "model": "tfidf",
            "device": "cpu",
            "seconds": time.perf_counter() - start,
            "unit": "full fit incl. C search",
        }
    )
    print(rows[-1])

    for device in devices:
        for name in NEURAL_MODELS:
            seconds = seconds_per_epoch(name, torch.device(device), encoded["train"], vocab_size)
            rows.append({"model": name, "device": device, "seconds": seconds, "unit": "epoch"})
            print(rows[-1])

    meta = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
    }
    table = pd.DataFrame([{**row, **meta} for row in rows])
    table.to_csv(args.output, index=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

"""Training loop with early stopping on validation accuracy."""

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agnews import config
from agnews.data import iterate_batches


@dataclass
class TrainConfig:
    batch_size: int = config.BATCH_SIZE
    learning_rate: float = config.LEARNING_RATE
    max_epochs: int = config.MAX_EPOCHS
    patience: int = config.PATIENCE
    grad_clip: float = config.GRAD_CLIP


@dataclass
class TrainResult:
    history: list[dict]
    best_epoch: int
    best_val_accuracy: float
    train_seconds: float


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def predict(model, input_ids, labels, device, batch_size=512):
    """Return (mean loss, predicted labels as a numpy array)."""
    model.eval()
    total_loss, predictions = 0.0, []
    for batch_ids, batch_labels in iterate_batches(input_ids, labels, batch_size):
        logits = model(batch_ids.to(device))
        total_loss += F.cross_entropy(logits, batch_labels.to(device), reduction="sum").item()
        predictions.append(logits.argmax(dim=1).cpu())
    return total_loss / len(labels), torch.cat(predictions).numpy()


def train_model(model, train, validation, device, seed, train_config=None, log=print):
    """Train with Adam, keep the weights from the epoch with the best validation accuracy,
    and stop after `patience` epochs without improvement.

    `train` and `validation` are (input_ids, labels) tensor pairs. The model is left on
    `device` with its best weights loaded.
    """
    cfg = train_config or TrainConfig()
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    history, train_seconds = [], 0.0
    best_accuracy, best_epoch, best_state, stale_epochs = -1.0, 0, None, 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        start = time.perf_counter()
        # Accumulate on-device so we don't force a device sync every step.
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        for input_ids, labels in iterate_batches(
            *train, cfg.batch_size, shuffle=True, generator=generator
        ):
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            loss_sum += loss.detach() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum()
        synchronize(device)
        epoch_seconds = time.perf_counter() - start
        train_seconds += epoch_seconds

        val_loss, val_predictions = predict(model, *validation, device)
        val_accuracy = float((val_predictions == validation[1].numpy()).mean())
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum.item() / len(train[1]),
                "train_accuracy": correct.item() / len(train[1]),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "epoch_seconds": epoch_seconds,
            }
        )
        log(
            f"  epoch {epoch:2d}  train_loss {history[-1]['train_loss']:.4f}  "
            f"val_loss {val_loss:.4f}  val_acc {val_accuracy:.4f}  ({epoch_seconds:.1f}s)"
        )

        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch, stale_epochs = val_accuracy, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return TrainResult(history, best_epoch, best_accuracy, train_seconds)

import numpy as np
import torch

from agnews.baselines import TfidfLogReg
from agnews.metrics import classification_metrics, parse_confusion
from agnews.models import build_model
from agnews.training import TrainConfig, predict, set_seed, train_model

CPU = torch.device("cpu")


def keyword_dataset(n, seed):
    """Class k is signalled by token 2 + k appearing somewhere among random filler tokens."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 4, (n,), generator=generator)
    ids = torch.randint(10, 40, (n, 8), generator=generator)
    position = torch.randint(0, 8, (n,), generator=generator)
    ids[torch.arange(n), position] = 2 + labels
    return ids, labels


def quiet(_message):
    pass


def test_fnn_learns_a_keyword_task_and_restores_best_epoch():
    set_seed(0)
    model = build_model("fnn", 40, embedding_dim=16, hidden_dim=32)
    config = TrainConfig(batch_size=32, learning_rate=1e-2, max_epochs=15, patience=3)
    result = train_model(
        model, keyword_dataset(512, 1), keyword_dataset(256, 2), CPU, 0, config, log=quiet
    )
    best = max(result.history, key=lambda epoch: epoch["val_accuracy"])
    assert result.best_epoch == best["epoch"]
    _, predictions = predict(model, *keyword_dataset(256, 2), CPU)
    val_accuracy = (predictions == keyword_dataset(256, 2)[1].numpy()).mean()
    assert val_accuracy == result.best_val_accuracy > 0.9


def test_training_is_reproducible_for_a_fixed_seed():
    def run():
        set_seed(3)
        model = build_model("lstm", 40, embedding_dim=8, hidden_dim=8)
        config = TrainConfig(batch_size=32, max_epochs=2)
        return train_model(
            model, keyword_dataset(128, 1), keyword_dataset(64, 2), CPU, 3, config, log=quiet
        ).history

    first, second = run(), run()
    assert [e["train_loss"] for e in first] == [e["train_loss"] for e in second]


def test_tfidf_baseline_learns_keyword_documents():
    rng = np.random.default_rng(0)
    topics = ["goal", "stock", "chip", "war"]

    def docs(n):
        labels = rng.integers(0, 4, n)
        return [[topics[y], "the", "news", "today"] for y in labels], labels

    train, val, test = docs(200), docs(50), docs(50)
    model = TfidfLogReg(c_grid=(1.0,)).fit(*train, *val)
    assert (model.predict(test[0]) == test[1]).mean() == 1.0


def test_classification_metrics_round_trip_the_confusion_matrix():
    y_true = np.array([0, 1, 2, 3, 3])
    y_pred = np.array([0, 1, 2, 2, 3])
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["test_accuracy"] == 0.8
    assert parse_confusion(metrics["confusion"])[3].tolist() == [0, 0, 1, 1]

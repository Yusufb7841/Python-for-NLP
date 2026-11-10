"""The four neural classifiers. They share embedding and hidden sizes so that the only
thing that differs between them is how a sequence of word vectors becomes one vector."""

import torch
import torch.nn as nn

from agnews.data import NUM_CLASSES, PAD_ID

NEURAL_MODELS = ("fnn", "rnn", "lstm", "attention")


class MeanPoolClassifier(nn.Module):
    """Bag-of-embeddings: average the word vectors, then a one-hidden-layer MLP.
    Ignores word order entirely."""

    def __init__(self, embedding, hidden_dim, dropout, num_classes=NUM_CLASSES):
        super().__init__()
        self.embedding = embedding
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids):
        mask = (input_ids != PAD_ID).unsqueeze(-1).float()
        summed = (self.embedding(input_ids) * mask).sum(dim=1)
        return self.classifier(summed / mask.sum(dim=1).clamp(min=1))


class RecurrentClassifier(nn.Module):
    """Vanilla RNN or LSTM; classifies from the hidden state at the last real token."""

    def __init__(self, embedding, hidden_dim, dropout, cell="lstm", num_classes=NUM_CLASSES):
        super().__init__()
        encoder_class = {"rnn": nn.RNN, "lstm": nn.LSTM}[cell]
        self.embedding = embedding
        self.encoder = encoder_class(embedding.embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        outputs, _ = self.encoder(self.embedding(input_ids))
        lengths = (input_ids != PAD_ID).sum(dim=1).clamp(min=1)
        rows = torch.arange(input_ids.size(0), device=input_ids.device)
        return self.classifier(self.dropout(outputs[rows, lengths - 1]))


class AttentionLSTMClassifier(nn.Module):
    """LSTM followed by additive attention pooling over all (non-padding) time steps."""

    def __init__(self, embedding, hidden_dim, dropout, num_classes=NUM_CLASSES):
        super().__init__()
        self.embedding = embedding
        self.encoder = nn.LSTM(embedding.embedding_dim, hidden_dim, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, return_attention=False):
        outputs, _ = self.encoder(self.embedding(input_ids))
        scores = self.attention(outputs).squeeze(-1)
        weights = scores.masked_fill(input_ids == PAD_ID, float("-inf")).softmax(dim=1)
        context = (weights.unsqueeze(-1) * outputs).sum(dim=1)
        logits = self.classifier(self.dropout(context))
        return (logits, weights) if return_attention else logits


def build_model(name, vocab_size, embedding_dim=100, hidden_dim=128, dropout=0.2, pretrained=None):
    embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
    if pretrained is not None:
        with torch.no_grad():
            embedding.weight.copy_(pretrained)
    if name == "fnn":
        return MeanPoolClassifier(embedding, hidden_dim, dropout)
    if name in ("rnn", "lstm"):
        return RecurrentClassifier(embedding, hidden_dim, dropout, cell=name)
    if name == "attention":
        return AttentionLSTMClassifier(embedding, hidden_dim, dropout)
    raise ValueError(f"Unknown model {name!r}; expected one of {NEURAL_MODELS}")


def count_parameters(model, include_embedding=True):
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (include_embedding or not name.startswith("embedding."))
    )

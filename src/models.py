import torch
import torch.nn as nn


class FNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=100, hidden_dim=128, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        mask = (input_ids != self.pad_idx).unsqueeze(-1)
        summed = (embedded * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = summed / lengths
        return self.classifier(pooled)


class RNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=100, hidden_dim=128, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, _ = self.rnn(embedded)
        lengths = (input_ids != self.pad_idx).sum(dim=1).clamp(min=1)
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        final_hidden = output[batch_indices, lengths - 1]
        return self.classifier(final_hidden)


class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=100, hidden_dim=128, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, _ = self.lstm(embedded)
        lengths = (input_ids != self.pad_idx).sum(dim=1).clamp(min=1)
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        final_hidden = output[batch_indices, lengths - 1]
        return self.classifier(final_hidden)


class AttentionLSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=100, hidden_dim=128, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_output, _ = self.lstm(embedded)
        attention_scores = self.attention(lstm_output).squeeze(-1)
        padding_mask = input_ids == self.pad_idx
        attention_scores = attention_scores.masked_fill(padding_mask, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)
        return self.classifier(context)


def get_model(
    model_name,
    vocab_size,
    num_classes,
    embedding_dim=100,
    hidden_dim=128,
    pad_idx=0,
):
    model_name = model_name.lower()

    if model_name == "fnn":
        return FNNTextClassifier(vocab_size, num_classes, embedding_dim, hidden_dim, pad_idx)
    if model_name == "rnn":
        return RNNTextClassifier(vocab_size, num_classes, embedding_dim, hidden_dim, pad_idx)
    if model_name == "lstm":
        return LSTMTextClassifier(vocab_size, num_classes, embedding_dim, hidden_dim, pad_idx)
    if model_name == "attention":
        return AttentionLSTMTextClassifier(vocab_size, num_classes, embedding_dim, hidden_dim, pad_idx)

    raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    input_ids = torch.randint(0, 5000, (8, 30))
    input_ids[:, -5:] = 0
    model_names = ["fnn", "rnn", "lstm", "attention"]

    for model_name in model_names:
        model = get_model(model_name, vocab_size=5000, num_classes=4)
        output = model(input_ids)
        print(model_name, output.shape)
        assert list(output.shape) == [8, 4]

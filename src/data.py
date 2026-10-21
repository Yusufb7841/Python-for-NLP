from collections import Counter

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def tokenize(text):
    return text.lower().split()


def build_vocabulary(texts, max_vocab_size=20000, min_frequency=2):
    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    vocabulary = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}

    for word, count in counter.most_common():
        if count < min_frequency:
            continue
        if len(vocabulary) >= max_vocab_size:
            break
        vocabulary[word] = len(vocabulary)

    return vocabulary


def text_to_ids(text, vocabulary, max_length):
    token_ids = [vocabulary.get(token, UNK_ID) for token in tokenize(text)]
    token_ids = token_ids[:max_length]

    if len(token_ids) < max_length:
        token_ids = token_ids + [PAD_ID] * (max_length - len(token_ids))

    return token_ids


class AGNewsDataset(Dataset):
    def __init__(self, examples, vocabulary, max_length):
        self.examples = examples
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        input_ids = text_to_ids(example["text"], self.vocabulary, self.max_length)
        label = example["label"]

        return {
            "example_index": index,
            "text": example["text"],
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


def create_data_loaders(
    batch_size=32,
    max_length=80,
    max_vocab_size=20000,
    min_frequency=2,
    validation_size=0.1,
    seed=42,
    train_limit=None,
    validation_limit=None,
    test_limit=None,
):
    dataset = load_dataset("ag_news")
    split_dataset = dataset["train"].train_test_split(
        test_size=validation_size,
        seed=seed,
        stratify_by_column="label",
    )

    train_data = split_dataset["train"]
    validation_data = split_dataset["test"]
    test_data = dataset["test"]

    if train_limit is not None:
        train_data = train_data.select(range(train_limit))
    if validation_limit is not None:
        validation_data = validation_data.select(range(validation_limit))
    if test_limit is not None:
        test_data = test_data.select(range(test_limit))

    vocabulary = build_vocabulary(
        train_data["text"],
        max_vocab_size=max_vocab_size,
        min_frequency=min_frequency,
    )

    train_dataset = AGNewsDataset(train_data, vocabulary, max_length)
    validation_dataset = AGNewsDataset(validation_data, vocabulary, max_length)
    test_dataset = AGNewsDataset(test_data, vocabulary, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, validation_loader, test_loader, vocabulary


if __name__ == "__main__":
    train_loader, validation_loader, test_loader, vocabulary = create_data_loaders(
        batch_size=8,
        max_length=30,
        max_vocab_size=5000,
        min_frequency=1,
        train_limit=200,
        validation_limit=40,
        test_limit=40,
    )

    train_batch = next(iter(train_loader))

    print("Vocabulary size:", len(vocabulary))
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))
    print("Test batches:", len(test_loader))
    print("Input batch shape:", train_batch["input_ids"].shape)
    print("Label batch shape:", train_batch["label"].shape)

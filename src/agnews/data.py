"""Loading, cleaning, tokenizing and encoding AG News."""

import html
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch
from sklearn.model_selection import train_test_split

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
NUM_CLASSES = len(LABEL_NAMES)
PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1

# AG News stores many HTML entities without the leading "&" (e.g. "#39;s", "quot;"),
# uses "\" as a line break, and sometimes contains escaped HTML tags.
_BARE_NUMERIC_ENTITY = re.compile(r"(?<!&)#(\d+);")
_BARE_NAMED_ENTITY = re.compile(r"(?<![&\w])(quot|amp|lt|gt);")
_HTML_TAG = re.compile(r"<[^>]*>")
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")


def clean_text(text):
    text = text.replace("\\", " ")
    text = _BARE_NUMERIC_ENTITY.sub(r"&#\1;", text)
    text = _BARE_NAMED_ENTITY.sub(r"&\1;", text)
    text = html.unescape(html.unescape(text))
    return _HTML_TAG.sub(" ", text)


def tokenize(text):
    """Lowercase word/number tokens; keeps dotted forms like "u.s" and "3.5" intact."""
    return _TOKEN.findall(clean_text(text).lower())


@dataclass
class Vocabulary:
    itos: list[str]
    stoi: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, token_lists, max_size=20000, min_freq=2):
        counts = Counter(token for tokens in token_lists for token in tokens)
        # Sort by (-count, token) so ties are broken deterministically.
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        words = [word for word, count in ranked if count >= min_freq]
        return cls([PAD_TOKEN, UNK_TOKEN] + words[: max_size - 2])

    def encode(self, tokens, max_length):
        ids = [self.stoi.get(token, UNK_ID) for token in tokens[:max_length]] or [UNK_ID]
        return ids + [PAD_ID] * (max_length - len(ids))


def encode_batch(token_lists, vocab, max_length):
    return torch.tensor(
        [vocab.encode(tokens, max_length) for tokens in token_lists], dtype=torch.long
    )


@dataclass
class Corpus:
    """The full AG News corpus, tokenized once and indexed by position."""

    train_texts: list[str]
    train_tokens: list[list[str]]
    train_labels: np.ndarray
    test_texts: list[str]
    test_tokens: list[list[str]]
    test_labels: np.ndarray


@lru_cache(maxsize=1)
def load_corpus():
    from datasets import load_dataset

    dataset = load_dataset("ag_news")
    train, test = dataset["train"], dataset["test"]
    return Corpus(
        train_texts=train["text"],
        train_tokens=[tokenize(text) for text in train["text"]],
        train_labels=np.asarray(train["label"]),
        test_texts=test["text"],
        test_tokens=[tokenize(text) for text in test["text"]],
        test_labels=np.asarray(test["label"]),
    )


def split_indices(labels, seed, validation_fraction=0.1, train_size=None):
    """Stratified train/validation split of the official training set.

    The validation set is always the full held-out fraction, so it does not shrink when
    `train_size` subsamples the training portion (as in the learning-curve runs).
    """
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices, test_size=validation_fraction, stratify=labels, random_state=seed
    )
    if train_size is not None and train_size < len(train_idx):
        train_idx, _ = train_test_split(
            train_idx, train_size=train_size, stratify=labels[train_idx], random_state=seed
        )
    return np.sort(train_idx), np.sort(val_idx)


def iterate_batches(input_ids, labels, batch_size, shuffle=False, generator=None):
    """Yield (input_ids, labels) batches, trimming trailing columns that are all padding."""
    order = (
        torch.randperm(len(labels), generator=generator) if shuffle else torch.arange(len(labels))
    )
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        batch_ids = input_ids[batch]
        width = int((batch_ids != PAD_ID).sum(dim=1).max())
        yield batch_ids[:, :width], labels[batch]

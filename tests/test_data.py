import numpy as np
import torch

from agnews.data import (
    PAD_ID,
    UNK_ID,
    Vocabulary,
    clean_text,
    encode_batch,
    iterate_batches,
    split_indices,
    tokenize,
)


def test_clean_text_repairs_bare_html_entities():
    assert clean_text("Wall Street #39;s rally") == "Wall Street 's rally"
    assert clean_text("said quot;hello quot;") == 'said "hello "'
    assert clean_text("Oil hits  #36;46 a barrel") == "Oil hits  $46 a barrel"


def test_clean_text_strips_escaped_tags_and_backslash_breaks():
    assert clean_text("&lt;b&gt;bold&lt;/b&gt; text") == " bold  text"
    assert clean_text("first line\\second line") == "first line second line"


def test_tokenize_lowercases_and_drops_punctuation():
    text = "Wall St. Bears Claw Back (Reuters) - Short-sellers, U.S. stocks rose 3.5%."
    assert tokenize(text) == [
        "wall", "st", "bears", "claw", "back", "reuters", "short", "sellers",
        "u.s", "stocks", "rose", "3.5",
    ]  # fmt: skip


def test_vocabulary_orders_by_frequency_and_applies_limits():
    docs = [["b", "a", "a"], ["a", "c", "b"], ["rare"]]
    vocab = Vocabulary.build(docs, max_size=10, min_freq=2)
    assert vocab.itos == ["<pad>", "<unk>", "a", "b"]
    assert len(Vocabulary.build(docs, max_size=3, min_freq=1)) == 3


def test_encode_pads_truncates_and_maps_unknowns():
    vocab = Vocabulary(["<pad>", "<unk>", "a", "b"])
    assert vocab.encode(["a", "zzz"], max_length=4) == [2, UNK_ID, PAD_ID, PAD_ID]
    assert vocab.encode(["a", "b", "a", "b", "a"], max_length=3) == [2, 3, 2]
    # An empty document still gets one real token so pooling never divides by zero.
    assert vocab.encode([], max_length=3) == [UNK_ID, PAD_ID, PAD_ID]
    assert encode_batch([["a"], ["b", "b"]], vocab, 2).tolist() == [[2, 0], [3, 3]]


def test_split_indices_is_stratified_disjoint_and_deterministic():
    labels = np.repeat(np.arange(4), 250)
    train, val = split_indices(labels, seed=0, validation_fraction=0.2)
    assert len(train) == 800 and len(val) == 200
    assert not set(train) & set(val)
    assert np.bincount(labels[val]).tolist() == [50, 50, 50, 50]
    assert np.array_equal(train, split_indices(labels, seed=0, validation_fraction=0.2)[0])
    assert not np.array_equal(train, split_indices(labels, seed=1, validation_fraction=0.2)[0])


def test_split_indices_subsamples_train_but_keeps_full_validation():
    labels = np.repeat(np.arange(4), 250)
    _, full_val = split_indices(labels, seed=0, validation_fraction=0.2)
    train, val = split_indices(labels, seed=0, validation_fraction=0.2, train_size=100)
    assert len(train) == 100
    assert np.bincount(labels[train]).tolist() == [25, 25, 25, 25]
    assert np.array_equal(val, full_val)


def test_iterate_batches_trims_padding_columns_and_covers_every_example():
    ids = torch.tensor([[5, 6, 0, 0], [7, 0, 0, 0], [8, 9, 4, 0]])
    labels = torch.tensor([0, 1, 2])
    batches = list(iterate_batches(ids, labels, batch_size=2))
    assert [b[0].shape for b in batches] == [(2, 2), (1, 3)]
    shuffled = iterate_batches(
        ids, labels, 2, shuffle=True, generator=torch.Generator().manual_seed(0)
    )
    assert sorted(torch.cat([b[1] for b in shuffled]).tolist()) == [0, 1, 2]

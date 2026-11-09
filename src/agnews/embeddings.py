"""Pretrained GloVe vectors: download once, then build an embedding matrix for a vocabulary."""

import argparse
import shutil
import urllib.request
import zipfile
from functools import lru_cache

import numpy as np
import torch

from agnews.config import GLOVE_PATH
from agnews.data import PAD_ID

GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"


def download_glove(path=GLOVE_PATH, url=GLOVE_URL):
    if path.exists():
        print(f"GloVe already present at {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = path.parent / "glove.6B.zip"
    print(f"Downloading {url} (~860 MB)...")
    urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as zf, zf.open(path.name) as src, open(path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    archive.unlink()
    print(f"Saved {path}")
    return path


@lru_cache(maxsize=2)
def read_glove(path=GLOVE_PATH):
    index, rows = {}, []
    with open(path, encoding="utf8") as file:
        for line in file:
            word, values = line.rstrip().split(" ", 1)
            index[word] = len(rows)
            rows.append(np.array(values.split(), dtype=np.float32))
    return index, np.stack(rows)


def glove_matrix(vocab, seed, path=GLOVE_PATH):
    """Embedding matrix for `vocab`: GloVe rows where available, random rows (matched to
    GloVe's scale) for the rest, zeros for padding. Returns (matrix, coverage)."""
    index, vectors = read_glove(path)
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, vectors.std(), size=(len(vocab), vectors.shape[1])).astype(np.float32)
    matrix[PAD_ID] = 0.0
    found = 0
    for vocab_id, word in enumerate(vocab.itos[2:], start=2):
        # Our tokenizer drops trailing dots ("u.s."), GloVe's keeps them.
        glove_id = index.get(word, index.get(word + "."))
        if glove_id is not None:
            matrix[vocab_id] = vectors[glove_id]
            found += 1
    return torch.from_numpy(matrix), found / max(len(vocab) - 2, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GloVe 6B 100d vectors.")
    parser.parse_args()
    download_glove()

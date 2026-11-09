"""Paths and the hyperparameters shared by every run. Nothing here is tuned per model."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
GLOVE_PATH = PROJECT_ROOT / "data" / "glove" / "glove.6B.100d.txt"

MAX_VOCAB_SIZE = 20000
MIN_TOKEN_FREQ = 2
MAX_LENGTH = 64  # covers 98% of articles in full (99th percentile is 71 tokens)
VALIDATION_FRACTION = 0.1

EMBEDDING_DIM = 100  # matches GloVe 6B 100d
HIDDEN_DIM = 128
DROPOUT = 0.2

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
PATIENCE = 3
GRAD_CLIP = 1.0

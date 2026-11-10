"""Non-neural reference point: TF-IDF over word unigrams + bigrams with logistic regression."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

C_GRID = (1.0, 4.0, 16.0)


def _identity(tokens):
    return tokens


class TfidfLogReg:
    """Fits on pre-tokenized documents; the regularization strength C is picked on the
    validation set, the same data the neural models use for early stopping."""

    def __init__(self, c_grid=C_GRID, max_iter=1000):
        self.c_grid = c_grid
        self.max_iter = max_iter
        self.vectorizer = TfidfVectorizer(
            tokenizer=_identity,
            preprocessor=_identity,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        self.classifier = None
        self.best_c = None
        self.best_val_accuracy = None

    def fit(self, train_tokens, train_labels, val_tokens, val_labels):
        train_features = self.vectorizer.fit_transform(train_tokens)
        val_features = self.vectorizer.transform(val_tokens)
        self.best_val_accuracy = -1.0
        for c in self.c_grid:
            classifier = LogisticRegression(C=c, max_iter=self.max_iter)
            classifier.fit(train_features, train_labels)
            accuracy = float(np.mean(classifier.predict(val_features) == val_labels))
            if accuracy > self.best_val_accuracy:
                self.classifier, self.best_c, self.best_val_accuracy = classifier, c, accuracy
        return self

    def predict(self, tokens):
        return self.classifier.predict(self.vectorizer.transform(tokens))

    @property
    def num_features(self):
        return len(self.vectorizer.vocabulary_)

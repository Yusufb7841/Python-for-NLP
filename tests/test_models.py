import pytest
import torch

from agnews.data import PAD_ID
from agnews.models import NEURAL_MODELS, build_model, count_parameters

VOCAB_SIZE = 50


def make_batch():
    torch.manual_seed(0)
    ids = torch.randint(2, VOCAB_SIZE, (3, 6))
    ids[0, 4:] = PAD_ID
    ids[1, 1:] = PAD_ID
    return ids


@pytest.mark.parametrize("name", NEURAL_MODELS)
def test_forward_returns_one_logit_per_class(name):
    model = build_model(name, VOCAB_SIZE, embedding_dim=8, hidden_dim=16)
    assert model(make_batch()).shape == (3, 4)


@pytest.mark.parametrize("name", NEURAL_MODELS)
def test_extra_trailing_padding_does_not_change_predictions(name):
    model = build_model(name, VOCAB_SIZE, embedding_dim=8, hidden_dim=16).eval()
    ids = make_batch()
    padded = torch.cat([ids, torch.full((3, 5), PAD_ID)], dim=1)
    torch.testing.assert_close(model(ids), model(padded))


def test_attention_weights_are_a_distribution_over_real_tokens():
    model = build_model("attention", VOCAB_SIZE, embedding_dim=8, hidden_dim=16).eval()
    ids = make_batch()
    _, weights = model(ids, return_attention=True)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))
    assert torch.all(weights[ids == PAD_ID] == 0)


def test_pretrained_vectors_are_copied_into_the_embedding():
    vectors = torch.randn(VOCAB_SIZE, 8)
    model = build_model("fnn", VOCAB_SIZE, embedding_dim=8, hidden_dim=16, pretrained=vectors)
    torch.testing.assert_close(model.embedding.weight.detach(), vectors)


def test_parameter_count_can_exclude_the_embedding_table():
    model = build_model("lstm", VOCAB_SIZE, embedding_dim=8, hidden_dim=16)
    assert count_parameters(model) - count_parameters(model, include_embedding=False) == 50 * 8


def test_unknown_model_name_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("transformer", VOCAB_SIZE)

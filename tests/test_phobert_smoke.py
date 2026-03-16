"""Offline smoke test for the training data/model interface."""

import os

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_PHOBERT_SMOKE") != "1",
    reason="set RUN_PHOBERT_SMOKE=1 after installing requirements-phobert.txt",
)
def test_tiny_roberta_forward_and_save(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import RobertaConfig, RobertaForSequenceClassification

    config = RobertaConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        num_labels=3,
    )
    model = RobertaForSequenceClassification(config)
    output = model(
        input_ids=torch.tensor([[0, 5, 6, 2]]),
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
        labels=torch.tensor([2]),
    )
    assert output.logits.shape == (1, 3)
    assert torch.isfinite(output.loss)
    model.save_pretrained(tmp_path)
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "model.safetensors").exists()

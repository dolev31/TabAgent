"""The e5 encoder fails with one clear message when its extra or its weights are missing."""

import sys

import pytest

from tabagent.tabschema.fitted import E5Encoder


def test_missing_embed_extra_is_one_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch now raises ImportError
    with pytest.raises(ImportError, match=r"tabagent\[embed\]"):
        E5Encoder()


def test_unavailable_weights_are_one_clear_error_and_hub_settings_are_restored(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from transformers.utils import logging as hf_logging

    def offline(*args, **kwargs):
        raise OSError("We couldn't connect to 'https://huggingface.co' to load the files")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", offline)
    before = (hf_logging.get_verbosity(), hf_logging.is_progress_bar_enabled())
    with pytest.raises(OSError, match="pair_embeddings=False") as err:
        E5Encoder()
    assert "couldn't connect" in str(err.value) and "130 MB" in str(err.value)
    assert (hf_logging.get_verbosity(), hf_logging.is_progress_bar_enabled()) == before

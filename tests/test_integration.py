"""
Integration tests for repr-audit using a lightweight test model.
"""

from __future__ import annotations

import numpy as np
import pytest

from repr_audit.auditor import RepresentationAuditor
from repr_audit.extractor import HiddenStateExtractor


@pytest.fixture(scope="module")
def tiny_model_name():
    # Standard lightweight HuggingFace test model (~50KB)
    return "hf-internal-testing/tiny-random-BertModel"


def test_extractor_extract(tiny_model_name):
    extractor = HiddenStateExtractor(tiny_model_name)
    sentences = ["This is a test sentence.", "Another sentence here."]
    states = extractor.extract(sentences)

    # 5 hidden layers -> 6 layer reps including embedding
    assert states.shape[0] == 6
    assert states.shape[1] == 2
    assert states.shape[2] == 32


def test_extractor_token_representations(tiny_model_name):
    extractor = HiddenStateExtractor(tiny_model_name)
    sentences = [
        "She sat on the river bank.",
        "He went to the money bank.",
    ]
    token_reps = extractor.extract_token_representations(sentences, "bank")
    # Shape: (num_layers+1, num_occurrences, hidden)
    assert token_reps.shape[0] == 6
    assert token_reps.shape[1] == 2
    assert token_reps.shape[2] == 32


def test_auditor_end_to_end(tiny_model_name):
    auditor = RepresentationAuditor(tiny_model_name, batch_size=4)
    sentences = [
        "The river bank was muddy and steep.",
        "She deposited checks at the bank.",
        "Central bank interest rates fluctuated.",
        "Fish were swimming near the bank.",
    ]
    token_sentences = {"bank": sentences}
    labels = np.array([1, 0, 0, 1])

    results = auditor.audit(
        sentences=sentences,
        token_sentences=token_sentences,
        labels=labels,
        keep_representations=True,
    )

    assert results.model_name == tiny_model_name
    assert results.n_sentences == 4
    assert len(results.anisotropy_per_layer) == 6
    assert len(results.uncentered_svd_ratio_per_layer) == 6
    assert len(results.self_similarity_per_layer) == 6
    assert len(results.silhouette_per_layer) == 6
    assert results.layer_representations is not None
    assert "Model" in results.summary()

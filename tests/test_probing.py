"""
Unit tests for diagnostic probing module.
"""

from __future__ import annotations

import numpy as np
import pytest

from repr_audit.probing import (
    compare_geometry_against_probing,
    evaluate_linear_classification_probe,
    evaluate_sts_similarity_probe,
    find_plateau_layer,
)


@pytest.fixture
def synthetic_probe_data():
    """Simulate 6 layers of representation for 60 samples with 3 classes."""
    rng = np.random.default_rng(123)
    n_layers = 6
    n_samples = 60
    d = 32

    # Let class 0, 1, 2 become progressively better separated with depth
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    layer_reps = []

    for l in range(n_layers):
        base = rng.standard_normal((n_samples, d)).astype(np.float32)
        # Add class signal that increases with layer
        signal_scale = l * 1.5
        base[:20, 0] += signal_scale
        base[20:40, 1] += signal_scale
        base[40:60, 2] += signal_scale
        layer_reps.append(base)

    return np.stack(layer_reps, axis=0), labels


def test_find_plateau_layer():
    # Curve rising to peak at layer 4
    scores = [0.2, 0.4, 0.8, 0.94, 0.95, 0.95]
    plateau = find_plateau_layer(scores, fraction_of_peak=0.95)
    # At layer 3 (0.94 / 0.95 = 0.989 >= 0.95), plateau reached
    assert plateau == 3


def test_linear_classification_probe(synthetic_probe_data):
    layer_reps, labels = synthetic_probe_data
    res = evaluate_linear_classification_probe(
        layer_representations=layer_reps,
        labels=labels,
        task_name="POS Probing",
        dimension="syntactic",
    )
    assert len(res.scores_per_layer) == 6
    # Early layers should have lower score than later layers
    assert res.scores_per_layer[-1] >= res.scores_per_layer[0]
    assert 0 <= res.peak_layer < 6
    assert 0 <= res.plateau_layer <= res.peak_layer
    summary_str = res.summary()
    assert "POS Probing" in summary_str


def test_sts_similarity_probe():
    rng = np.random.default_rng(42)
    n_layers = 5
    n_pairs = 30
    d = 16

    a = rng.standard_normal((n_layers, n_pairs, d)).astype(np.float32)
    b = a.copy() + 0.1 * rng.standard_normal((n_layers, n_pairs, d)).astype(np.float32)
    gold = np.linspace(0.1, 5.0, n_pairs)

    res = evaluate_sts_similarity_probe(a, b, gold)
    assert len(res.scores_per_layer) == 5
    assert -1.0 <= res.peak_score <= 1.0


def test_compare_geometry_against_probing(synthetic_probe_data):
    layer_reps, labels = synthetic_probe_data
    p1 = evaluate_linear_classification_probe(
        layer_reps, labels, task_name="Task1", dimension="syntactic"
    )
    p2 = evaluate_linear_classification_probe(
        layer_reps, labels, task_name="Task2", dimension="semantic"
    )

    comparison = compare_geometry_against_probing(
        sod_layer=3,
        cka_elbow_layer=2,
        max_drift_layer=1,
        probing_results=[p1, p2],
    )
    assert "mae_sod_vs_probes" in comparison
    assert "mae_cka_vs_probes" in comparison
    assert len(comparison["detailed_comparisons"]) == 2

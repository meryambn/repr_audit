"""
Comprehensive unit tests for representation geometry metrics in repr-audit.

Run with: pytest tests/ -v --cov=repr_audit
"""

from __future__ import annotations

import numpy as np
import pytest

from repr_audit.metrics import (
    compute_anisotropy,
    compute_cka_elbow,
    compute_layer_drift,
    compute_linear_cka,
    compute_max_drift_layer,
    compute_mev,
    compute_self_similarity,
    compute_semantic_onset_depth,
    compute_uncentered_svd_ratio,
    evaluate_sod_sensitivity,
)
from repr_audit.results import AuditResults

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def random_reps() -> np.ndarray:
    """Random representations: (50, 64) — 50 sentences, 64-dim hidden."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 64)).astype(np.float32)


@pytest.fixture
def layer_reps() -> np.ndarray:
    """Simulated layer representations: (13, 50, 64) — 13 layers, 50 sentences."""
    rng = np.random.default_rng(0)
    layers = []
    for l in range(13):
        base = rng.standard_normal((50, 64))
        if l < 4:
            dominant = rng.standard_normal((1, 64))
            base += 5.0 * dominant
        layers.append(base)
    return np.stack(layers, axis=0).astype(np.float32)


@pytest.fixture
def dummy_results(layer_reps) -> AuditResults:
    """A minimal AuditResults instance for testing."""
    n_layers = layer_reps.shape[0]
    anisotropy = [compute_anisotropy(layer_reps[l]) for l in range(n_layers)]
    mev = [compute_mev(layer_reps[l]) for l in range(n_layers)]
    svd_ratio = [compute_uncentered_svd_ratio(layer_reps[l]) for l in range(n_layers)]
    drift = compute_layer_drift(layer_reps)
    sod = compute_semantic_onset_depth(layer_reps)
    cka_elbow = compute_cka_elbow(layer_reps)
    max_drift = compute_max_drift_layer(layer_reps)
    return AuditResults(
        model_name="test-model",
        n_sentences=50,
        n_layers=n_layers - 1,
        anisotropy_per_layer=anisotropy,
        mev_per_layer=mev,
        uncentered_svd_ratio_per_layer=svd_ratio,
        layer_drift_cosine=drift["cosine_distance"],
        layer_drift_cka=drift["linear_cka"],
        semantic_onset_depth=sod,
        cka_elbow_layer=cka_elbow,
        max_drift_layer=max_drift,
    )


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------


class TestAnisotropy:
    def test_output_range(self, random_reps):
        score = compute_anisotropy(random_reps)
        assert -1.0 <= score <= 1.0

    def test_random_is_near_zero(self, random_reps):
        score = compute_anisotropy(random_reps)
        assert abs(score) < 0.15, f"Expected near-zero, got {score:.4f}"

    def test_constant_direction_is_one(self):
        v = np.ones((50, 32), dtype=np.float32)
        score = compute_anisotropy(v)
        assert score > 0.99

    def test_exact_computation_small_n(self):
        # Exact vs sample test
        v = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        # Pairwise cos: (0, 1) -> 0; (0, 2) -> 1/sqrt(2); (1, 2) -> 1/sqrt(2)
        expected = float((0.0 + 1 / np.sqrt(2) + 1 / np.sqrt(2)) / 3.0)
        score = compute_anisotropy(v, exact_threshold=10)
        assert abs(score - expected) < 1e-5

    def test_confidence_interval(self, random_reps):
        score, (ci_low, ci_high) = compute_anisotropy(random_reps, return_ci=True)
        assert ci_low <= score <= ci_high

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_anisotropy(np.ones((1, 32)))


class TestUncenteredSVD:
    def test_output_range(self, random_reps):
        score = compute_uncentered_svd_ratio(random_reps)
        assert 0.0 <= score <= 1.0

    def test_rank1_matrix(self):
        v = np.ones((40, 32), dtype=np.float32)
        score = compute_uncentered_svd_ratio(v)
        assert abs(score - 1.0) < 1e-5

    def test_cone_detection_vs_centered_pca(self):
        """When vectors have a huge common mean but isotropic spread around that mean:
        uncentered SVD ratio is near 1.0 (cone concentration), while centered PCA MEV is low."""
        rng = np.random.default_rng(42)
        # Random noise with dim 64
        noise = rng.standard_normal((100, 64)).astype(np.float32)
        # Add massive shift along 1 direction
        mean_direction = np.zeros((1, 64), dtype=np.float32)
        mean_direction[0, 0] = 50.0  # huge offset
        cone_data = noise + mean_direction

        uncentered_ratio = compute_uncentered_svd_ratio(cone_data)
        centered_mev = compute_mev(cone_data)

        assert uncentered_ratio > 0.90, f"Uncentered SVD should detect cone: got {uncentered_ratio}"
        assert centered_mev < 0.20, f"Centered PCA MEV removes mean: got {centered_mev}"


class TestSelfSimilarity:
    def test_output_range(self):
        rng = np.random.default_rng(1)
        token_reps = {
            "bank": rng.standard_normal((10, 64)).astype(np.float32),
            "run": rng.standard_normal((8, 64)).astype(np.float32),
        }
        score = compute_self_similarity(token_reps)
        assert -1.0 <= score <= 1.0

    def test_identical_reps_give_one(self):
        v = np.ones((5, 32), dtype=np.float32)
        score = compute_self_similarity({"test": v})
        assert abs(score - 1.0) < 1e-5

    def test_single_occurrence_skipped(self):
        rng = np.random.default_rng(2)
        token_reps = {
            "single": rng.standard_normal((1, 32)).astype(np.float32),
            "multi": rng.standard_normal((5, 32)).astype(np.float32),
        }
        score = compute_self_similarity(token_reps)
        assert -1.0 <= score <= 1.0

    def test_all_single_raises(self):
        token_reps = {"a": np.ones((1, 32), dtype=np.float32)}
        with pytest.raises(ValueError, match="No token"):
            compute_self_similarity(token_reps)


class TestLinearCKAAndDrift:
    def test_identical_representations_give_one(self, random_reps):
        cka = compute_linear_cka(random_reps, random_reps)
        assert abs(cka - 1.0) < 1e-5

    def test_orthogonal_representations_low(self):
        rng = np.random.default_rng(99)
        x = rng.standard_normal((200, 30))
        y = rng.standard_normal((200, 30))
        cka = compute_linear_cka(x, y)
        assert cka < 0.2

    def test_layer_drift_computes(self, layer_reps):
        drift = compute_layer_drift(layer_reps)
        assert "cosine_distance" in drift
        assert "linear_cka" in drift
        assert len(drift["cosine_distance"]) == layer_reps.shape[0] - 1

    def test_cka_elbow_returns_layer(self, layer_reps):
        elbow = compute_cka_elbow(layer_reps, min_similarity=0.5)
        assert isinstance(elbow, int)


class TestSemanticOnsetDepth:
    def test_returns_int(self, layer_reps):
        sod = compute_semantic_onset_depth(layer_reps)
        assert isinstance(sod, int)

    def test_constant_returns_zero(self):
        rng = np.random.default_rng(5)
        base = rng.standard_normal((30, 64)).astype(np.float32)
        layer_reps = np.stack([base] * 13, axis=0)
        sod = compute_semantic_onset_depth(layer_reps)
        assert sod == 0

    def test_sensitivity_evaluation(self, layer_reps):
        grid = evaluate_sod_sensitivity(layer_reps, thresholds=[0.3, 0.5, 0.8], windows=[1, 2])
        assert "window_1" in grid
        assert "window_2" in grid
        assert "thresh_0.50" in grid["window_1"]


# ---------------------------------------------------------------------------
# AuditResults tests
# ---------------------------------------------------------------------------


class TestAuditResults:
    def test_summary_runs(self, dummy_results):
        s = dummy_results.summary()
        assert "Model" in s
        assert "test-model" in s
        assert "SOD (Hypothesis)" in s
        assert "Max SVD1 Cone" in s

    def test_json_roundtrip(self, dummy_results, tmp_path):
        path = tmp_path / "results.json"
        dummy_results.to_json(path)
        loaded = AuditResults.from_json(path)
        assert loaded.model_name == dummy_results.model_name
        assert loaded.semantic_onset_depth == dummy_results.semantic_onset_depth
        assert len(loaded.uncentered_svd_ratio_per_layer) == len(
            dummy_results.uncentered_svd_ratio_per_layer
        )

"""
Systematic diagnostic probing suite for transformer layer representations.

Evaluates linear probes across 4 distinct linguistic dimensions:
1. Syntactic  : Part-of-speech (POS) tagging
2. Entity     : Named Entity Recognition (NER)
3. Semantic   : Sentence / topic classification
4. Sentence-Level Semantics: Semantic Textual Similarity (STS) correlation

Maps layer-wise probe trajectories to test whether geometric stabilization
(Semantic Onset Depth / CKA elbow / Max Drift) corresponds to linguistic usability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


@dataclass
class ProbingResults:
    """Stores layer-wise diagnostic probe performance curves and plateau markers."""

    task_name: str
    dimension: Literal["syntactic", "entity", "semantic", "sentence_semantics"]
    metric_name: str
    scores_per_layer: list[float]
    peak_layer: int
    peak_score: float
    plateau_layer: int

    def summary(self) -> str:
        return (
            f"Task: {self.task_name:25s} [{self.dimension:18s}] | "
            f"Peak Layer: {self.peak_layer:2d} ({self.metric_name}={self.peak_score:.3f}) | "
            f"Plateau Layer: {self.plateau_layer:2d}"
        )


def find_plateau_layer(
    scores: list[float],
    fraction_of_peak: float = 0.95,
) -> int:
    """Find the earliest layer where probe performance reaches >= fraction_of_peak * max_score.

    Parameters
    ----------
    scores:
        List of layer-wise metric scores.
    fraction_of_peak:
        Threshold ratio (default 0.95, i.e. within 5% of peak performance).

    Returns
    -------
    int: Layer index (0-indexed).
    """
    if not scores:
        return -1
    max_score = max(scores)
    if max_score <= 0.0:
        return int(np.argmax(scores))

    threshold = fraction_of_peak * max_score
    for layer, score in enumerate(scores):
        if score >= threshold:
            return layer
    return int(np.argmax(scores))


def evaluate_linear_classification_probe(
    layer_representations: np.ndarray,
    labels: np.ndarray,
    task_name: str = "classification",
    dimension: Literal["syntactic", "entity", "semantic"] = "semantic",
    metric_name: str = "Accuracy",
    test_size: float = 0.3,
    random_state: int = 42,
    use_ridge: bool = True,
) -> ProbingResults:
    """Train a linear probe at each layer to evaluate representation utility.

    Parameters
    ----------
    layer_representations:
        Shape (n_layers + 1, n_samples, hidden_size).
    labels:
        Integer class labels of shape (n_samples,).
    """
    labels = np.asarray(labels)
    n_layers_total = layer_representations.shape[0]

    # Stratified split to ensure class balance across splits
    idx_train, idx_test = train_test_split(
        np.arange(len(labels)),
        test_size=test_size,
        random_state=random_state,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )
    y_train, y_test = labels[idx_train], labels[idx_test]

    scores_per_layer: list[float] = []

    for l in range(n_layers_total):
        X = layer_representations[l]
        X_train, X_test = X[idx_train], X[idx_test]

        if use_ridge:
            clf = RidgeClassifier(alpha=1.0, random_state=random_state)
        else:
            clf = LogisticRegression(max_iter=500, random_state=random_state, C=1.0)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if len(np.unique(labels)) > 2:
            score = float(f1_score(y_test, y_pred, average="weighted"))
        else:
            score = float((y_test == y_pred).mean())

        scores_per_layer.append(score)

    peak_layer = int(np.argmax(scores_per_layer))
    peak_score = float(scores_per_layer[peak_layer])
    plateau_layer = find_plateau_layer(scores_per_layer)

    return ProbingResults(
        task_name=task_name,
        dimension=dimension,
        metric_name=metric_name,
        scores_per_layer=scores_per_layer,
        peak_layer=peak_layer,
        peak_score=peak_score,
        plateau_layer=plateau_layer,
    )


def evaluate_sts_similarity_probe(
    layer_reps_a: np.ndarray,
    layer_reps_b: np.ndarray,
    gold_scores: np.ndarray,
) -> ProbingResults:
    """Evaluate sentence-level semantic representation alignment via STS Spearman correlation.

    Parameters
    ----------
    layer_reps_a:
        Shape (n_layers + 1, n_pairs, hidden_size) for sentence 1.
    layer_reps_b:
        Shape (n_layers + 1, n_pairs, hidden_size) for sentence 2.
    gold_scores:
        Continuous human similarity ratings of shape (n_pairs,).
    """
    gold_scores = np.asarray(gold_scores)
    n_layers_total = layer_reps_a.shape[0]
    rhos: list[float] = []

    for l in range(n_layers_total):
        # L2 normalize
        a = layer_reps_a[l] / np.maximum(
            np.linalg.norm(layer_reps_a[l], axis=1, keepdims=True), 1e-12
        )
        b = layer_reps_b[l] / np.maximum(
            np.linalg.norm(layer_reps_b[l], axis=1, keepdims=True), 1e-12
        )
        cos_sims = (a * b).sum(axis=1)

        rho, _ = spearmanr(cos_sims, gold_scores)
        rhos.append(float(0.0 if np.isnan(rho) else rho))

    peak_layer = int(np.argmax(rhos))
    peak_score = float(rhos[peak_layer])
    plateau_layer = find_plateau_layer(rhos)

    return ProbingResults(
        task_name="STS Benchmark",
        dimension="sentence_semantics",
        metric_name="Spearman rho",
        scores_per_layer=rhos,
        peak_layer=peak_layer,
        peak_score=peak_score,
        plateau_layer=plateau_layer,
    )


def compare_geometry_against_probing(
    sod_layer: int,
    cka_elbow_layer: int,
    max_drift_layer: int,
    probing_results: list[ProbingResults],
) -> dict[str, object]:
    """Quantitatively compare geometric inflection markers against diagnostic probe plateaus.

    Tests the hypothesis: Does geometric stabilization (SOD/CKA) correspond to
    functional linguistic plateau depth?
    """
    comparison_table = []
    plateau_layers = []

    for p in probing_results:
        diff_sod = p.plateau_layer - sod_layer
        diff_cka = p.plateau_layer - cka_elbow_layer
        diff_drift = p.plateau_layer - max_drift_layer
        plateau_layers.append(p.plateau_layer)

        comparison_table.append(
            {
                "task": p.task_name,
                "dimension": p.dimension,
                "probe_peak_layer": p.peak_layer,
                "probe_plateau_layer": p.plateau_layer,
                "sod_layer": sod_layer,
                "delta_sod": diff_sod,
                "cka_elbow_layer": cka_elbow_layer,
                "delta_cka": diff_cka,
                "max_drift_layer": max_drift_layer,
                "delta_drift": diff_drift,
            }
        )

    avg_plateau = float(np.mean(plateau_layers)) if plateau_layers else 0.0
    mean_abs_error_sod = float(np.mean([abs(row["delta_sod"]) for row in comparison_table]))
    mean_abs_error_cka = float(np.mean([abs(row["delta_cka"]) for row in comparison_table]))
    mean_abs_error_drift = float(np.mean([abs(row["delta_drift"]) for row in comparison_table]))

    return {
        "sod_marker": sod_layer,
        "cka_elbow_marker": cka_elbow_layer,
        "max_drift_marker": max_drift_layer,
        "average_probe_plateau": avg_plateau,
        "mae_sod_vs_probes": mean_abs_error_sod,
        "mae_cka_vs_probes": mean_abs_error_cka,
        "mae_drift_vs_probes": mean_abs_error_drift,
        "detailed_comparisons": comparison_table,
    }

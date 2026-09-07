"""
AuditResults: structured container for all metric outputs.

Designed to be serialisable to JSON/CSV and directly plottable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class AuditResults:
    """Container for all metrics produced by RepresentationAuditor.

    Attributes
    ----------
    model_name:
        HuggingFace model identifier.
    n_sentences:
        Number of sentences used.
    n_layers:
        Number of transformer layers (excluding embedding).
    anisotropy_per_layer:
        Anisotropy score at each layer (including embedding layer at index 0).
    mev_per_layer:
        Maximum Explainable Variance (centered PCA) at each layer.
    uncentered_svd_ratio_per_layer:
        Uncentered SVD energy ratio of top singular value at each layer.
    layer_drift_cosine:
        Consecutive-layer cosine distance.
    layer_drift_cka:
        Consecutive-layer linear Centered Kernel Alignment (CKA) similarity.
    self_similarity_per_layer:
        Average intra-token cosine similarity at each layer.
        Empty list if no token_representations were provided.
    silhouette_per_layer:
        Silhouette coefficient at each layer.
        Empty list if no labels were provided.
    semantic_onset_depth:
        The layer index where anisotropy changes stabilize (-1 if not found).
    cka_elbow_layer:
        The layer where consecutive CKA similarity plateaus (-1 if not found).
    max_drift_layer:
        The layer with maximum representation shift from its predecessor.
    layer_representations:
        Raw hidden states. Shape (n_layers+1, n_sentences, hidden_size).
        None by default to save memory; set keep_representations=True in auditor.
    """

    model_name: str
    n_sentences: int
    n_layers: int
    anisotropy_per_layer: list[float]
    mev_per_layer: list[float]
    uncentered_svd_ratio_per_layer: list[float] = field(default_factory=list)
    layer_drift_cosine: list[float] = field(default_factory=list)
    layer_drift_cka: list[float] = field(default_factory=list)
    self_similarity_per_layer: list[float] = field(default_factory=list)
    silhouette_per_layer: list[float] = field(default_factory=list)
    semantic_onset_depth: int = -1
    cka_elbow_layer: int = -1
    max_drift_layer: int = -1
    layer_representations: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def layer_indices(self) -> list[int]:
        """Layer indices from 0 (embedding) to n_layers."""
        return list(range(len(self.anisotropy_per_layer)))

    @property
    def avg_anisotropy(self) -> float:
        """Mean anisotropy across all layers."""
        return float(np.mean(self.anisotropy_per_layer))

    @property
    def min_anisotropy_layer(self) -> int:
        """Layer with the lowest anisotropy (most isotropic)."""
        return int(np.argmin(self.anisotropy_per_layer))

    @property
    def max_mev_layer(self) -> int:
        """Layer with the highest MEV."""
        return int(np.argmax(self.mev_per_layer))

    @property
    def max_svd_ratio_layer(self) -> int:
        """Layer with highest directional cone concentration."""
        if not self.uncentered_svd_ratio_per_layer:
            return 0
        return int(np.argmax(self.uncentered_svd_ratio_per_layer))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain dict (excluding raw representations)."""
        d = asdict(self)
        d.pop("layer_representations", None)
        return d

    def to_json(self, path: str | Path) -> None:
        """Save results to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> AuditResults:
        """Load results from a JSON file."""
        data = json.loads(Path(path).read_text())
        data.pop("layer_representations", None)
        return cls(**data)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Model            : {self.model_name}",
            f"Sentences        : {self.n_sentences}",
            f"Layers           : {self.n_layers}",
            f"Avg Anisotropy   : {self.avg_anisotropy:.4f}",
            f"SOD (Hypothesis) : {self.semantic_onset_depth} "
            f"({'not found' if self.semantic_onset_depth == -1 else 'layer ' + str(self.semantic_onset_depth)})",
            f"CKA Elbow Layer  : {self.cka_elbow_layer}",
            f"Max Drift Layer  : {self.max_drift_layer}",
            f"Most Isotropic   : layer {self.min_anisotropy_layer}",
            f"Max MEV Layer    : layer {self.max_mev_layer}",
        ]
        if self.uncentered_svd_ratio_per_layer:
            lines.append(
                f"Max SVD1 Cone    : layer {self.max_svd_ratio_layer} "
                f"({self.uncentered_svd_ratio_per_layer[self.max_svd_ratio_layer]:.4f})"
            )
        if self.self_similarity_per_layer:
            avg_ss = float(np.mean(self.self_similarity_per_layer))
            lines.append(f"Avg Self-Sim     : {avg_ss:.4f}")
        return "\n".join(lines)

    def plot(self, **kwargs) -> None:
        """Shortcut to AuditPlotter(self).plot(**kwargs)."""
        from repr_audit.plotting import AuditPlotter

        AuditPlotter(self).plot(**kwargs)

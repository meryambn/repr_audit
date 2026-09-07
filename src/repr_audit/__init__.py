"""
repr-audit: Auditing transformer representation geometry.

Computes layer-wise anisotropy, uncentered SVD concentration, self-similarity,
MEV, CKA alignment, layer drift, and validates the Semantic Onset Depth hypothesis.
"""

from repr_audit.auditor import RepresentationAuditor
from repr_audit.extractor import HiddenStateExtractor
from repr_audit.metrics import (
    compute_anisotropy,
    compute_cka_elbow,
    compute_layer_drift,
    compute_linear_cka,
    compute_max_drift_layer,
    compute_mev,
    compute_self_similarity,
    compute_semantic_onset_depth,
    compute_silhouette,
    compute_uncentered_svd_ratio,
    evaluate_sod_sensitivity,
)
from repr_audit.plotting import AuditPlotter
from repr_audit.probing import (
    ProbingResults,
    compare_geometry_against_probing,
    evaluate_linear_classification_probe,
    evaluate_sts_similarity_probe,
)
from repr_audit.results import AuditResults

__version__ = "0.2.0"
__all__ = [
    "RepresentationAuditor",
    "HiddenStateExtractor",
    "AuditPlotter",
    "AuditResults",
    "ProbingResults",
    "compute_anisotropy",
    "compute_uncentered_svd_ratio",
    "compute_self_similarity",
    "compute_mev",
    "compute_linear_cka",
    "compute_layer_drift",
    "compute_cka_elbow",
    "compute_max_drift_layer",
    "compute_silhouette",
    "compute_semantic_onset_depth",
    "evaluate_sod_sensitivity",
    "evaluate_linear_classification_probe",
    "evaluate_sts_similarity_probe",
    "compare_geometry_against_probing",
]

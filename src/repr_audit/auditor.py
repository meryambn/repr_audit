"""
RepresentationAuditor: the primary public API for repr-audit.

Orchestrates extraction, metric computation, and result packaging.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from repr_audit.extractor import ExtractionConfig, HiddenStateExtractor
from repr_audit.metrics import (
    compute_anisotropy,
    compute_cka_elbow,
    compute_layer_drift,
    compute_max_drift_layer,
    compute_mev,
    compute_self_similarity,
    compute_semantic_onset_depth,
    compute_silhouette,
    compute_uncentered_svd_ratio,
)
from repr_audit.results import AuditResults


class RepresentationAuditor:
    """High-level API for auditing transformer representation geometry.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (e.g. "bert-base-uncased", "gpt2", "meta-llama/Llama-3.2-1B").
    pooling:
        Token pooling strategy. If None, auto-detected from model family.
    batch_size:
        Sentences per forward pass.
    device:
        Torch device. Defaults to CUDA if available.
    max_length:
        Maximum tokenisation length.
    padding_side:
        Padding side ("left" or "right"). Defaults to "left" for decoders and "right" for encoders.

    Examples
    --------
    Basic usage::

        auditor = RepresentationAuditor("bert-base-uncased")
        results = auditor.audit(sentences=["The bank was steep.", "The bank approved the loan."])
        print(results.summary())
        results.plot()
    """

    def __init__(
        self,
        model_name: str,
        pooling: Literal["mean", "cls", "last_token"] | None = None,
        batch_size: int = 32,
        device: str | None = None,
        max_length: int = 128,
        padding_side: Literal["left", "right"] | None = None,
    ) -> None:
        self.model_name = model_name

        config_kwargs: dict = dict(
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            padding_side=padding_side,
        )
        if device is not None:
            config_kwargs["device"] = device

        config_kwargs["pooling"] = pooling or "mean"
        self._extractor = HiddenStateExtractor(ExtractionConfig(**config_kwargs))

        # Auto-detect pooling based on model family if not explicitly given
        if pooling is None:
            family = self._extractor.detect_model_family()
            if family == "decoder":
                self._extractor.config.pooling = "last_token"

    def audit(
        self,
        sentences: list[str],
        token_sentences: dict[str, list[str]] | None = None,
        labels: np.ndarray | None = None,
        keep_representations: bool = False,
        anisotropy_samples: int = 1000,
        sod_threshold: float = 0.50,
        sod_window: int = 2,
    ) -> AuditResults:
        """Run a full geometry audit on the given sentences.

        Parameters
        ----------
        sentences:
            List of raw text strings.
        token_sentences:
            Optional dict mapping target words to lists of sentences containing them.
            Isolated token embeddings are extracted and evaluated for contextual self-similarity.
        labels:
            Optional integer labels for silhouette computation.
        keep_representations:
            If True, store raw hidden states in results.layer_representations.
        anisotropy_samples:
            Number of sample pairs if dataset exceeds exact calculation threshold.
        sod_threshold:
            Threshold factor for hypothesized Semantic Onset Depth.
        sod_window:
            Window size for hypothesized Semantic Onset Depth.

        Returns
        -------
        AuditResults with all computed representation geometry metrics.
        """
        # Step 1: Extract layer hidden states in a single batched pass
        layer_reps = self._extractor.extract(sentences)
        n_layers_with_emb = layer_reps.shape[0]
        n_layers = n_layers_with_emb - 1

        # Step 2: Layer-wise representation geometry
        anisotropy = [
            compute_anisotropy(layer_reps[l], n_samples=anisotropy_samples)
            for l in range(n_layers_with_emb)
        ]
        mev = [compute_mev(layer_reps[l]) for l in range(n_layers_with_emb)]
        svd_ratio = [compute_uncentered_svd_ratio(layer_reps[l]) for l in range(n_layers_with_emb)]

        # Step 3: Layer drift and CKA alignment
        drift = compute_layer_drift(layer_reps)
        cka_elbow = compute_cka_elbow(layer_reps)
        max_drift_l = compute_max_drift_layer(layer_reps)

        # Step 4: Semantic Onset Depth (hypothesized)
        sod = compute_semantic_onset_depth(layer_reps, threshold=sod_threshold, window=sod_window)

        # Step 5: Self-similarity via actual target-token subword extraction
        self_sim_per_layer: list[float] = []
        if token_sentences is not None:
            # Extract target token representations once per token
            token_reps_by_word: dict[str, np.ndarray] = {}
            for word, ctxs in token_sentences.items():
                token_reps = self._extractor.extract_token_representations(ctxs, word)
                if token_reps.shape[1] >= 2:
                    token_reps_by_word[word] = token_reps

            if token_reps_by_word:
                for l in range(n_layers_with_emb):
                    layer_dict = {word: token_reps_by_word[word][l] for word in token_reps_by_word}
                    self_sim_per_layer.append(compute_self_similarity(layer_dict))

        # Step 6: Silhouette score (if labels provided)
        silhouette_per_layer: list[float] = []
        if labels is not None:
            labels = np.asarray(labels)
            for l in range(n_layers_with_emb):
                silhouette_per_layer.append(compute_silhouette(layer_reps[l], labels))

        return AuditResults(
            model_name=self.model_name,
            n_sentences=len(sentences),
            n_layers=n_layers,
            anisotropy_per_layer=anisotropy,
            mev_per_layer=mev,
            uncentered_svd_ratio_per_layer=svd_ratio,
            layer_drift_cosine=drift["cosine_distance"],
            layer_drift_cka=drift["linear_cka"],
            self_similarity_per_layer=self_sim_per_layer,
            silhouette_per_layer=silhouette_per_layer,
            semantic_onset_depth=sod,
            cka_elbow_layer=cka_elbow,
            max_drift_layer=max_drift_l,
            layer_representations=layer_reps if keep_representations else None,
        )

    def audit_multiple_models(
        self,
        model_names: list[str],
        sentences: list[str],
        **audit_kwargs,
    ) -> list[AuditResults]:
        """Audit multiple models on the same sentences."""
        results = []
        for name in model_names:
            auditor = RepresentationAuditor(
                name,
                batch_size=self._extractor.config.batch_size,
                device=self._extractor.config.device,
                max_length=self._extractor.config.max_length,
            )
            results.append(auditor.audit(sentences, **audit_kwargs))
        return results

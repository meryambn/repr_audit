"""
Plotting utilities for AuditResults and diagnostic probing analysis.

All plots are publication-quality (300 DPI, clean theme, consistent typography)
and can be saved to disk or displayed interactively.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

if TYPE_CHECKING:
    from repr_audit.probing import ProbingResults
    from repr_audit.results import AuditResults

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
PALETTE = sns.color_palette("tab10")


class AuditPlotter:
    """Generates publication-quality plots from AuditResults."""

    def __init__(self, results: AuditResults) -> None:
        self.r = results

    def plot(
        self,
        metrics: list[str] | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot representation geometry metrics in a multi-panel figure.

        Options: "anisotropy", "svd_ratio", "mev", "drift", "cka", "self_similarity", "silhouette"
        """
        available = ["anisotropy"]
        if self.r.uncentered_svd_ratio_per_layer:
            available.append("svd_ratio")
        available.append("mev")
        if self.r.layer_drift_cka:
            available.append("cka")
        if self.r.layer_drift_cosine:
            available.append("drift")
        if self.r.self_similarity_per_layer:
            available.append("self_similarity")
        if self.r.silhouette_per_layer:
            available.append("silhouette")

        if metrics is None:
            metrics = available

        n_plots = len(metrics)
        fig, axes = plt.subplots(1, n_plots, figsize=(4.8 * n_plots, 4.0), constrained_layout=True)
        if n_plots == 1:
            axes = [axes]

        fig.suptitle(
            f"Representation Geometry Audit — {self.r.model_name}", fontsize=13, fontweight="bold"
        )

        plot_fn = {
            "anisotropy": self._plot_anisotropy,
            "svd_ratio": self._plot_svd_ratio,
            "mev": self._plot_mev,
            "cka": self._plot_cka,
            "drift": self._plot_drift,
            "self_similarity": self._plot_self_similarity,
            "silhouette": self._plot_silhouette,
        }

        for ax, metric in zip(axes, metrics):
            if metric in plot_fn:
                plot_fn[metric](ax)

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def _layer_x(self) -> list[int]:
        return self.r.layer_indices

    def _plot_anisotropy(self, ax: plt.Axes) -> None:
        x = self._layer_x()
        y = self.r.anisotropy_per_layer
        ax.plot(x, y, marker="o", color=PALETTE[0], linewidth=2, markersize=4, label="Anisotropy")

        sod = self.r.semantic_onset_depth
        if sod >= 0:
            ax.axvline(
                sod,
                color="crimson",
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                label=f"SOD (Hypothesis) = {sod}",
            )
        if self.r.cka_elbow_layer >= 0:
            ax.axvline(
                self.r.cka_elbow_layer,
                color="teal",
                linestyle=":",
                linewidth=1.5,
                alpha=0.85,
                label=f"CKA Elbow = {self.r.cka_elbow_layer}",
            )

        ax.set_title("Anisotropy per Layer")
        ax.set_xlabel("Layer Depth")
        ax.set_ylabel("Avg Cosine Similarity")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="best")

    def _plot_svd_ratio(self, ax: plt.Axes) -> None:
        x = self._layer_x()
        y = self.r.uncentered_svd_ratio_per_layer
        ax.plot(x, y, marker="v", color="darkmagenta", linewidth=2, markersize=4)
        ax.set_title("Uncentered SVD Ratio (Top Singular Value)")
        ax.set_xlabel("Layer Depth")
        ax.set_ylabel(r"Cone Concentration ($\sigma_1^2 / \sum \sigma_i^2$)")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_ylim(0, 1.05)

    def _plot_mev(self, ax: plt.Axes) -> None:
        x = self._layer_x()
        y = self.r.mev_per_layer
        ax.plot(x, y, marker="s", color=PALETTE[1], linewidth=2, markersize=4)
        ax.set_title("MEV per Layer (Centered PCA)")
        ax.set_xlabel("Layer Depth")
        ax.set_ylabel("Explained Variance Ratio")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_ylim(0, 1.05)

    def _plot_cka(self, ax: plt.Axes) -> None:
        ckas = self.r.layer_drift_cka
        x = list(range(1, len(ckas) + 1))
        ax.plot(x, ckas, marker="D", color="teal", linewidth=2, markersize=4)
        if self.r.cka_elbow_layer >= 0:
            ax.axvline(
                self.r.cka_elbow_layer,
                color="teal",
                linestyle=":",
                linewidth=1.5,
                label=f"Elbow = {self.r.cka_elbow_layer}",
            )
            ax.legend(fontsize=8)
        ax.set_title("Layer-to-Layer CKA Alignment")
        ax.set_xlabel("Transition (Layer $l \to l+1$)")
        ax.set_ylabel("Linear CKA Similarity")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_ylim(0, 1.05)

    def _plot_drift(self, ax: plt.Axes) -> None:
        dists = self.r.layer_drift_cosine
        x = list(range(1, len(dists) + 1))
        ax.plot(x, dists, marker="x", color="darkorange", linewidth=2, markersize=5)
        ax.set_title("Layer-to-Layer Cosine Drift")
        ax.set_xlabel("Transition (Layer $l \to l+1$)")
        ax.set_ylabel(r"Cosine Distance ($1 - \cos$)")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_ylim(0, max(max(dists) * 1.2, 0.5) if dists else 1.0)

    def _plot_self_similarity(self, ax: plt.Axes) -> None:
        x = self._layer_x()
        y = self.r.self_similarity_per_layer
        ax.plot(x, y, marker="^", color=PALETTE[2], linewidth=2, markersize=4)
        ax.set_title("Token Self-Similarity per Layer")
        ax.set_xlabel("Layer Depth")
        ax.set_ylabel("Avg Intra-Token Cosine Sim")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_ylim(-0.05, 1.05)

    def _plot_silhouette(self, ax: plt.Axes) -> None:
        x = self._layer_x()
        y = self.r.silhouette_per_layer
        ax.plot(x, y, marker="D", color=PALETTE[3], linewidth=2, markersize=4)
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_title("Silhouette Score per Layer")
        ax.set_xlabel("Layer Depth")
        ax.set_ylabel("Silhouette Coefficient")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_ylim(-1.05, 1.05)


def plot_model_comparison(
    results_list: list[AuditResults],
    metric: str = "anisotropy",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Compare a single representation metric across multiple model architectures."""
    metric_map = {
        "anisotropy": ("anisotropy_per_layer", "Anisotropy (Avg Cosine Similarity)"),
        "svd_ratio": (
            "uncentered_svd_ratio_per_layer",
            r"Uncentered SVD Energy Ratio ($\sigma_1^2 / \sum \sigma_i^2$)",
        ),
        "mev": ("mev_per_layer", "Centered PCA MEV"),
        "self_similarity": ("self_similarity_per_layer", "Token Self-Similarity"),
        "silhouette": ("silhouette_per_layer", "Silhouette Score"),
    }

    if metric not in metric_map:
        raise ValueError(f"Unknown metric: {metric!r}. Choose from {list(metric_map)}")

    attr, ylabel = metric_map[metric]
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)

    for i, r in enumerate(results_list):
        y = getattr(r, attr)
        if not y:
            continue
        x = r.layer_indices
        x_norm = [xi / max(x) for xi in x] if max(x) > 0 else x
        ax.plot(
            x_norm,
            y,
            label=r.model_name.split("/")[-1],
            color=PALETTE[i % len(PALETTE)],
            linewidth=2.2,
            marker="o",
            markersize=3.5,
        )

    ax.set_title(f"Cross-Architecture Comparison: {ylabel}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Normalized Layer Depth ($l / L$)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9, ncol=2)
    ax.set_ylim(-0.05, 1.05)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_probing_vs_geometry(
    probing_results: list[ProbingResults],
    audit_results: AuditResults,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Multi-task publication figure mapping layer-wise probe accuracy vs geometric markers."""
    n_tasks = len(probing_results)
    fig, axes = plt.subplots(1, n_tasks, figsize=(4.8 * n_tasks, 4.2), constrained_layout=True)
    if n_tasks == 1:
        axes = [axes]

    fig.suptitle(
        f"Linguistic Probing Trajectories vs Geometric Markers [{audit_results.model_name}]",
        fontsize=13,
        fontweight="bold",
    )

    sod = audit_results.semantic_onset_depth
    cka_elbow = audit_results.cka_elbow_layer
    max_drift = audit_results.max_drift_layer

    for ax, p in zip(axes, probing_results):
        x = list(range(len(p.scores_per_layer)))
        ax.plot(
            x,
            p.scores_per_layer,
            marker="o",
            linewidth=2.2,
            color=PALETTE[0],
            label=f"{p.metric_name}",
        )

        # Highlight geometric markers
        if sod >= 0:
            ax.axvline(
                sod,
                color="crimson",
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                label=f"SOD = {sod}",
            )
        if cka_elbow >= 0:
            ax.axvline(
                cka_elbow,
                color="teal",
                linestyle=":",
                linewidth=1.5,
                alpha=0.85,
                label=f"CKA Elbow = {cka_elbow}",
            )
        if max_drift >= 0:
            ax.axvline(
                max_drift,
                color="darkorange",
                linestyle="-.",
                linewidth=1.2,
                alpha=0.7,
                label=f"Max Drift = {max_drift}",
            )

        # Highlight probe plateau
        ax.axvline(
            p.plateau_layer,
            color="black",
            linestyle="-",
            linewidth=1.2,
            alpha=0.5,
            label=f"Probe Plateau = {p.plateau_layer}",
        )

        ax.set_title(f"{p.task_name}\n({p.dimension})", fontsize=11)
        ax.set_xlabel("Layer Depth")
        ax.set_ylabel(p.metric_name)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=7.5, loc="lower right")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig

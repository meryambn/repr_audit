"""
Generates publication-quality figures (300 DPI) from empirical benchmark results.

Outputs saved to `figures/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
PALETTE = sns.color_palette("tab10")


def generate_all_figures(
    benchmark_json_path: str = "benchmark_results/benchmark_data.json",
    figures_dir: str = "figures",
) -> None:
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    with open(benchmark_json_path, "r") as f:
        data = json.load(f)

    models_data = data["models"]

    # -----------------------------------------------------------------------
    # Figure 1: Cross-Architecture Representation Geometry (Anisotropy & SVD)
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for i, (name, m_info) in enumerate(models_data.items()):
        short_name = name.split("/")[-1]
        n_layers = m_info["n_layers"]
        x = np.linspace(0, 1, n_layers + 1)
        aniso = m_info["metrics"]["anisotropy_per_layer"]
        svd = m_info["metrics"]["uncentered_svd_ratio_per_layer"]

        ax1.plot(x, aniso, label=short_name, color=PALETTE[i], linewidth=2.2, marker="o", markersize=3.5)
        ax2.plot(x, svd, label=short_name, color=PALETTE[i], linewidth=2.2, marker="s", markersize=3.5)

    ax1.set_title("Global Anisotropy (Average Pairwise Cosine)", fontweight="bold")
    ax1.set_xlabel("Normalized Layer Depth ($l / L$)")
    ax1.set_ylabel("Anisotropy")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=9)

    ax2.set_title("Directional Cone Concentration (Uncentered SVD Ratio)", fontweight="bold")
    ax2.set_xlabel("Normalized Layer Depth ($l / L$)")
    ax2.set_ylabel("Fraction of Energy ($\sigma_1^2 / \sum \sigma_i^2$)")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)

    fig1_path = Path(figures_dir) / "fig1_cross_model_geometry.png"
    fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {fig1_path}")

    # -----------------------------------------------------------------------
    # Figure 2 & 3: Probing Trajectories vs Geometric Markers (BERT & GPT-2)
    # -----------------------------------------------------------------------
    for target_model in ["bert-base-uncased", "gpt2"]:
        if target_model not in models_data:
            continue
        m_info = models_data[target_model]
        probing_data = m_info["probing"]
        sod = m_info["sod_layer"]
        cka = m_info["cka_elbow_layer"]
        drift = m_info["max_drift_layer"]

        tasks = list(probing_data.keys())
        n_tasks = len(tasks)
        fig, axes = plt.subplots(1, n_tasks, figsize=(4.8 * n_tasks, 4.0), constrained_layout=True)
        if n_tasks == 1:
            axes = [axes]

        fig.suptitle(
            f"Linguistic Probing Trajectories vs Geometric Markers [{target_model}]",
            fontsize=13,
            fontweight="bold",
        )

        for ax, task_name in zip(axes, tasks):
            t_info = probing_data[task_name]
            scores = t_info["scores_per_layer"]
            x = list(range(len(scores)))

            ax.plot(x, scores, marker="o", linewidth=2.2, color=PALETTE[0], label=f"Probe ({t_info['metric_name']})")

            # Markers
            if sod >= 0:
                ax.axvline(sod, color="crimson", linestyle="--", linewidth=1.5, alpha=0.85, label=f"SOD = {sod}")
            if cka >= 0:
                ax.axvline(cka, color="teal", linestyle=":", linewidth=1.5, alpha=0.85, label=f"CKA Elbow = {cka}")
            if drift >= 0:
                ax.axvline(drift, color="darkorange", linestyle="-.", linewidth=1.2, alpha=0.7, label=f"Max Drift = {drift}")

            ax.axvline(t_info["plateau_layer"], color="black", linestyle="-", linewidth=1.2, alpha=0.5,
                       label=f"Plateau = {t_info['plateau_layer']}")

            ax.set_title(f"{task_name}\n({t_info['dimension']})", fontsize=11)
            ax.set_xlabel("Layer Depth")
            ax.set_ylabel(t_info["metric_name"])
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.legend(fontsize=7.5, loc="lower right")

        safe_name = target_model.replace("/", "_")
        fig_path = Path(figures_dir) / f"fig2_probing_vs_geometry_{safe_name}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {fig_path}")

    # -----------------------------------------------------------------------
    # Figure 4: SOD Threshold Sensitivity Heatmap
    # -----------------------------------------------------------------------
    if "bert-base-uncased" in models_data:
        m_info = models_data["bert-base-uncased"]
        grid = m_info["sod_sensitivity"]
        windows = list(grid.keys())
        threshs = list(grid[windows[0]].keys())

        matrix = np.zeros((len(windows), len(threshs)))
        for r, w in enumerate(windows):
            for c, t in enumerate(threshs):
                matrix[r, c] = grid[w][t]

        fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".0f",
            cmap="YlGnBu",
            xticklabels=[t.replace("thresh_", "") for t in threshs],
            yticklabels=[w.replace("window_", "w=") for w in windows],
            ax=ax,
            cbar_kws={"label": "Predicted Onset Layer"},
        )
        ax.set_title("Semantic Onset Depth (SOD) Sensitivity Grid [bert-base-uncased]", fontweight="bold")
        ax.set_xlabel("Threshold Multiplier ($\Delta < \\alpha \cdot \Delta_{\max}$)")
        ax.set_ylabel("Consecutive Window Size")

        fig4_path = Path(figures_dir) / "fig3_sod_sensitivity_heatmap.png"
        fig.savefig(fig4_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {fig4_path}")

    # -----------------------------------------------------------------------
    # Figure 5: Token-Level Self-Similarity Across Layers
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for i, (name, m_info) in enumerate(models_data.items()):
        self_sim = m_info["metrics"]["self_similarity_per_layer"]
        if self_sim:
            short_name = name.split("/")[-1]
            x = np.linspace(0, 1, len(self_sim))
            ax.plot(x, self_sim, label=short_name, color=PALETTE[i], linewidth=2.2, marker="^", markersize=3.5)

    ax.set_title("Token Contextualization: Intra-Token Self-Similarity across Layers", fontweight="bold")
    ax.set_xlabel("Normalized Layer Depth ($l / L$)")
    ax.set_ylabel("Intra-Token Cosine Similarity (Lower = More Contextualized)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)

    fig5_path = Path(figures_dir) / "fig4_token_self_similarity.png"
    fig.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {fig5_path}")


if __name__ == "__main__":
    generate_all_figures()

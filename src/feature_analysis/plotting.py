"""Plotting helpers for feature-analysis reports."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_rank_matrix(report: dict) -> tuple[list[str], list[str], list[list[float]]]:
    methods = list(report.get("per_method", {}).keys())
    feature_names = list(report.get("dataset", {}).get("feature_names", []))

    if not methods or not feature_names:
        return methods, feature_names, []

    by_method: list[list[float]] = []
    for method_name in methods:
        rows = report["per_method"][method_name]["ranking"]
        rank_map = {row["feature"]: float(row["mean_rank"]) for row in rows}
        by_method.append([rank_map.get(name, float("nan")) for name in feature_names])
    return methods, feature_names, by_method


def _resolve_relationship_features(
    report: dict,
    *,
    top_n: int,
) -> list[str]:
    feature_names = list(report.get("dataset", {}).get("feature_names", []))
    consensus = report.get("consensus", {}) or {}
    selected = list(consensus.get("selected", []))
    borda_order = list(consensus.get("borda_order", feature_names))

    ordered: list[str] = []
    seen: set[str] = set()
    for name in selected + borda_order:
        if name not in seen and name in feature_names:
            ordered.append(name)
            seen.add(name)
        if len(ordered) >= top_n:
            break
    return ordered


def _sample_indices(n_rows: int, max_points: int) -> list[int]:
    if max_points <= 0 or n_rows <= max_points:
        return list(range(n_rows))
    # Evenly spaced deterministic sampling keeps plots reproducible.
    step = (n_rows - 1) / max(max_points - 1, 1)
    return list(sorted({int(round(i * step)) for i in range(max_points)}))


def save_feature_analysis_plots(
    report: dict,
    output_dir: str | Path,
    *,
    data=None,
    top_n: int = 12,
    relationship_top_n: int = 6,
    sample_max: int = 5000,
    dpi: int = 150,
) -> list[str]:
    """Generate non-fatal PNG summaries for a feature-analysis report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available; skipping feature-analysis plots.")
        return saved

    # Import here so plotting remains optional for callers that only need JSON.
    from matplotlib import ticker

    feature_names = list(report.get("dataset", {}).get("feature_names", []))
    consensus = report.get("consensus", {}) or {}
    selected = set(consensus.get("selected", []))
    borda_order = list(consensus.get("borda_order", feature_names))
    borda_score_map = dict(zip(feature_names, consensus.get("borda_score", [])))
    stability_map = dict(zip(feature_names, consensus.get("mean_stability", [])))

    # 1. Borda consensus ranking.
    try:
        order = borda_order[:top_n]
        scores = [borda_score_map[name] for name in order]
        colors = ["tab:orange" if name in selected else "steelblue" for name in order]

        fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(order) + 1.5)))
        y = np.arange(len(order))
        ax.barh(y, scores, color=colors, edgecolor="black", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.set_xlabel("Borda Score (lower is better)")
        ax.set_title("Consensus Ranking")
        ax.invert_yaxis()
        for i, score in enumerate(scores):
            ax.text(score, i, f" {score:.2f}", va="center", fontsize=8)
        fig.tight_layout()
        path = out / "consensus_borda.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(str(path))
        plt.close(fig)
    except Exception as exc:
        logger.warning("Could not generate consensus_borda.png: %s", exc)

    # 2. Mean stability plot.
    try:
        order = sorted(feature_names, key=lambda name: stability_map.get(name, 0.0), reverse=True)[:top_n]
        values = [float(stability_map.get(name, 0.0)) for name in order]
        colors = ["tab:orange" if name in selected else "seagreen" for name in order]

        fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(order) + 1.5)))
        y = np.arange(len(order))
        ax.barh(y, values, color=colors, edgecolor="black", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Mean Stability")
        ax.set_title("Feature Stability Across Methods/Folds")
        ax.invert_yaxis()
        for i, val in enumerate(values):
            ax.text(min(val + 0.02, 0.98), i, f"{val:.2f}", va="center", fontsize=8)
        fig.tight_layout()
        path = out / "selection_stability.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(str(path))
        plt.close(fig)
    except Exception as exc:
        logger.warning("Could not generate selection_stability.png: %s", exc)

    # 3. Method-vs-feature rank heatmap.
    try:
        methods, matrix_features, rank_matrix = _extract_rank_matrix(report)
        if methods and matrix_features and rank_matrix:
            order = borda_order[:top_n]
            feature_idx = [matrix_features.index(name) for name in order]
            data = np.array(rank_matrix, dtype=float)[:, feature_idx]

            fig_w = max(8, 0.55 * len(order) + 2)
            fig_h = max(4.5, 0.5 * len(methods) + 2)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            im = ax.imshow(data, aspect="auto", cmap="viridis_r")
            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels(order, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(methods)))
            ax.set_yticklabels(methods)
            ax.set_title("Mean Rank by Method")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Mean Rank (lower is better)")

            if len(order) <= 14:
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", color="white", fontsize=7)

            fig.tight_layout()
            path = out / "method_rank_heatmap.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            saved.append(str(path))
            plt.close(fig)
    except Exception as exc:
        logger.warning("Could not generate method_rank_heatmap.png: %s", exc)

    # 4. Baseline metric comparison.
    try:
        baseline = report.get("baseline", {}) or {}
        per_model = baseline.get("per_model", {}) or {}
        valid_models = [
            (name, stats)
            for name, stats in per_model.items()
            if "r2_mean" in stats and "rmse_mean" in stats
        ]
        if valid_models:
            model_names = [name for name, _ in valid_models]
            r2_vals = [stats["r2_mean"] for _, stats in valid_models]
            r2_err = [stats["r2_std"] for _, stats in valid_models]
            rmse_vals = [stats["rmse_mean"] for _, stats in valid_models]
            rmse_err = [stats["rmse_std"] for _, stats in valid_models]

            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
            x = np.arange(len(model_names))

            axes[0].bar(x, r2_vals, yerr=r2_err, color="steelblue", edgecolor="black", alpha=0.85, capsize=4)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(model_names)
            axes[0].set_ylabel("R^2")
            axes[0].set_title("Baseline R^2")

            axes[1].bar(x, rmse_vals, yerr=rmse_err, color="indianred", edgecolor="black", alpha=0.85, capsize=4)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(model_names)
            axes[1].set_ylabel("RMSE")
            axes[1].set_title("Baseline RMSE")

            path = out / "baseline_metrics.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            saved.append(str(path))
            plt.close(fig)
    except Exception as exc:
        logger.warning("Could not generate baseline_metrics.png: %s", exc)

    # 5. Input-output coherence summary.
    try:
        if data is not None and getattr(data, "X", None) is not None and getattr(data, "y", None) is not None:
            order = _resolve_relationship_features(report, top_n=relationship_top_n)
            feat_idx = {name: i for i, name in enumerate(feature_names)}

            # Pull method-level relevance scores where available.
            mi_map: dict[str, float] = {}
            f_map: dict[str, float] = {}
            for row in report.get("per_method", {}).get("mutual_info", {}).get("ranking", []):
                mi_map[row["feature"]] = float(row["mean_score"])
            for row in report.get("per_method", {}).get("f_regression", {}).get("ranking", []):
                f_map[row["feature"]] = float(row["mean_score"])

            pearson_vals: list[float] = []
            mi_vals: list[float] = []
            f_vals: list[float] = []
            for name in order:
                col = np.asarray(data.X[:, feat_idx[name]], dtype=float)
                y = np.asarray(data.y, dtype=float)
                if np.std(col) < 1e-12 or np.std(y) < 1e-12:
                    pearson = 0.0
                else:
                    pearson = float(np.corrcoef(col, y)[0, 1])
                pearson_vals.append(pearson)
                mi_vals.append(float(mi_map.get(name, 0.0)))
                f_vals.append(float(f_map.get(name, 0.0)))

            fig, axes = plt.subplots(1, 3, figsize=(14, max(4.5, 0.45 * len(order) + 1.5)), constrained_layout=True)
            y_pos = np.arange(len(order))

            axes[0].barh(y_pos, pearson_vals, color="slateblue", edgecolor="black", alpha=0.85)
            axes[0].set_yticks(y_pos)
            axes[0].set_yticklabels(order)
            axes[0].invert_yaxis()
            axes[0].set_xlim(-1.0, 1.0)
            axes[0].set_xlabel("Pearson r")
            axes[0].set_title("Linear Coherence")

            axes[1].barh(y_pos, mi_vals, color="darkseagreen", edgecolor="black", alpha=0.85)
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels([])
            axes[1].invert_yaxis()
            axes[1].set_xlabel("Mutual Information")
            axes[1].set_title("Nonlinear Dependence")

            axes[2].barh(y_pos, f_vals, color="peru", edgecolor="black", alpha=0.85)
            axes[2].set_yticks(y_pos)
            axes[2].set_yticklabels([])
            axes[2].invert_yaxis()
            axes[2].set_xlabel("F-statistic")
            axes[2].set_title("Univariate Strength")
            axes[2].xaxis.set_major_locator(ticker.MaxNLocator(5))

            path = out / "feature_output_coherence.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            saved.append(str(path))
            plt.close(fig)
    except Exception as exc:
        logger.warning("Could not generate feature_output_coherence.png: %s", exc)

    # 6. Direct feature-target relationship panels.
    try:
        if data is not None and getattr(data, "X", None) is not None and getattr(data, "y", None) is not None:
            order = _resolve_relationship_features(report, top_n=relationship_top_n)
            feat_idx = {name: i for i, name in enumerate(feature_names)}
            idx = _sample_indices(len(data.y), sample_max)
            y = np.asarray(data.y[idx], dtype=float)

            n_panels = len(order)
            if n_panels:
                n_cols = 2
                n_rows = int(np.ceil(n_panels / n_cols))
                fig, axes = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(12, max(4.0, 3.2 * n_rows)),
                    constrained_layout=True,
                )
                axes_arr = np.atleast_1d(axes).reshape(n_rows, n_cols)

                for ax, name in zip(axes_arr.flat, order):
                    x = np.asarray(data.X[idx, feat_idx[name]], dtype=float)
                    uniq = np.unique(np.round(x, decimals=8))
                    if len(uniq) <= 3 and set(np.round(uniq).astype(int)).issubset({0, 1}):
                        groups = [y[x < 0.5], y[x >= 0.5]]
                        ax.boxplot(groups, labels=["0", "1"], patch_artist=True)
                        ax.set_xlabel(name)
                    else:
                        hb = ax.hexbin(x, y, gridsize=35, mincnt=1, cmap="viridis")
                        fig.colorbar(hb, ax=ax, label="count")
                        ax.set_xlabel(name)
                    ax.set_ylabel(report.get("dataset", {}).get("target_name", "target"))
                    ax.set_title(name)

                for ax in axes_arr.flat[n_panels:]:
                    ax.axis("off")

                path = out / "feature_target_relationships.png"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                saved.append(str(path))
                plt.close(fig)
    except Exception as exc:
        logger.warning("Could not generate feature_target_relationships.png: %s", exc)

    return saved

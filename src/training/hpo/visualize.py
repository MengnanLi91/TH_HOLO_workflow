"""Non-fatal Optuna visualization helpers."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def save_study_plots(study, output_dir: str | Path) -> list[str]:
    """Generate and save standard Optuna plots.  Non-fatal on errors.

    Returns list of saved file paths (may be empty if matplotlib or
    Optuna visualization is not available).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
            plot_slice,
        )
    except ImportError:
        logger.warning(
            "optuna.visualization.matplotlib not available. "
            "Install matplotlib for HPO plots."
        )
        return saved

    plots = [
        ("optimization_history", plot_optimization_history),
        ("param_importances", plot_param_importances),
        ("parallel_coordinate", plot_parallel_coordinate),
        ("slice_plot", plot_slice),
    ]

    for name, plot_fn in plots:
        try:
            ax = plot_fn(study)
            fig = ax.figure if hasattr(ax, "figure") else ax
            path = out / f"{name}.png"
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            saved.append(str(path))
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as exc:
            logger.warning("Could not generate %s: %s", name, exc)

    # Export trials CSV
    try:
        df = study.trials_dataframe()
        csv_path = out / "trials.csv"
        df.to_csv(str(csv_path), index=False)
        saved.append(str(csv_path))
    except Exception as exc:
        logger.warning("Could not export trials CSV: %s", exc)

    return saved

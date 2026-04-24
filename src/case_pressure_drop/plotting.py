"""Optional plotting helpers for case-level pressure-drop evaluation."""

from pathlib import Path

import numpy as np


def save_prediction_plots(
    *,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_dir: str | Path,
    feature_names: list[str] | None = None,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=np.float64)
    saved: list[str] = []
    feature_label = ", ".join(feature_names or [])
    if len(feature_label) > 60:
        feature_label = feature_label[:57] + "..."

    parity_path = output_dir / "parity_plot.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    max_val = float(max([float(y_true.max())] + [float(np.max(pred)) for pred in predictions.values()]))
    ax.plot([0.0, max_val], [0.0, max_val], linestyle="--", color="black", linewidth=1.0)
    for model_name, pred in predictions.items():
        ax.scatter(
            y_true,
            pred,
            s=28,
            alpha=0.75,
            label=model_name,
        )
    ax.set_xlabel("True delta_p_case [Pa]")
    ax.set_ylabel("Predicted delta_p_case [Pa]")
    ax.set_title("Pressure-Drop Parity")
    if feature_label:
        fig.suptitle(f"Features: {feature_label}", fontsize=10, y=0.98)
    ax.legend()
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96) if feature_label else None)
    fig.savefig(parity_path, dpi=160)
    plt.close(fig)
    saved.append(str(parity_path))

    residual_path = output_dir / "residual_plot.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    for model_name, pred in predictions.items():
        residual = np.asarray(pred, dtype=np.float64) - y_true
        ax.scatter(
            y_true,
            residual,
            s=28,
            alpha=0.75,
            label=model_name,
        )
    ax.set_xlabel("True delta_p_case [Pa]")
    ax.set_ylabel("Residual [Pa]")
    ax.set_title("Pressure-Drop Residuals")
    if feature_label:
        fig.suptitle(f"Features: {feature_label}", fontsize=10, y=0.98)
    ax.legend()
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96) if feature_label else None)
    fig.savefig(residual_path, dpi=160)
    plt.close(fig)
    saved.append(str(residual_path))

    return saved

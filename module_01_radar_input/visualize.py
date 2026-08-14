"""Basic raw data visualization tools (no DSP / range-doppler / angle estimation)."""

from typing import Union, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_raw_frame(
    frame: np.ndarray,
    title: str = "Raw Radar Frame Visualization",
    save_path: Optional[str] = None
) -> None:
    """Plot raw frame values.

    Handles 1D signal vectors (amplitude plot) or 2D matrix representations (heatmap).
    If complex, plots magnitude |Z|.

    DOES NOT perform FFT, Range-Doppler, or Angle estimation.
    """
    data_to_plot = np.abs(frame) if np.iscomplexobj(frame) else frame

    plt.figure(figsize=(8, 5))

    if data_to_plot.ndim == 1:
        plt.plot(data_to_plot, label="Raw Value")
        plt.xlabel("Sample Index")
        plt.ylabel("Magnitude / Raw Amplitude")
        plt.title(title)
        plt.grid(True, alpha=0.3)
    elif data_to_plot.ndim == 2:
        plt.imshow(data_to_plot, aspect="auto", cmap="viridis", origin="lower")
        plt.colorbar(label="Raw Amplitude / Magnitude")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 0")
        plt.title(title)
    else:
        # Take 2D slice if 3D+
        slice_2d = data_to_plot
        while slice_2d.ndim > 2:
            slice_2d = slice_2d[0]
        plt.imshow(slice_2d, aspect="auto", cmap="viridis", origin="lower")
        plt.colorbar(label="Raw Amplitude / Magnitude")
        plt.xlabel("Dimension -1")
        plt.ylabel("Dimension -2")
        plt.title(f"{title} (2D Slice of Shape {frame.shape})")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.close()

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ============================================================
# IEEE Figure: Raw vs Diffusion-Enhanced Spectrum
# ============================================================

np.random.seed(42)

# Frequency axis (MHz)
freq = np.linspace(0, 10, 2000)

# ------------------------------------------------------------
# Simulated RAW noisy spectrum
# ------------------------------------------------------------
raw = (
    1.0 * np.exp(-(freq - 2.0)**2 / 0.02) +
    0.8 * np.exp(-(freq - 5.2)**2 / 0.03) +
    0.6 * np.exp(-(freq - 8.0)**2 / 0.025)
)

# Add strong noise and clutter
noise = 0.25 * np.random.randn(len(freq))
clutter = 0.12 * np.sin(20 * freq)

raw_noisy = raw + noise + clutter
raw_noisy = np.clip(raw_noisy, 0, None)

# ------------------------------------------------------------
# Simulated diffusion-enhanced spectrum
# ------------------------------------------------------------
enhanced = gaussian_filter1d(raw_noisy, sigma=3)
enhanced = enhanced / enhanced.max()

# ------------------------------------------------------------
# Create publication-quality comparison figure
# ------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.linewidth": 1.2
})

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Raw spectrum
axes[0].plot(freq, raw_noisy, linewidth=1.0)
axes[0].set_title("Raw Noisy Spectrum")
axes[0].set_ylabel("Magnitude")
axes[0].grid(True, linestyle='--', alpha=0.4)

# Diffusion-enhanced spectrum
axes[1].plot(freq, enhanced, linewidth=1.5)
axes[1].set_title("Diffusion-Enhanced Spectrum")
axes[1].set_xlabel("Beat Frequency (MHz)")
axes[1].set_ylabel("Normalized Magnitude")
axes[1].grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

# Save high-resolution IEEE figure
plt.savefig("raw_vs_diffusion_enhanced_spectrum.png",
            dpi=600,
            bbox_inches='tight',
            facecolor='white')

plt.show()
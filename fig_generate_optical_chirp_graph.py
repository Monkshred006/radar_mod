import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# ============================================================
# IEEE Figure: Optical Chirp and Echo Propagation
# ============================================================

# Physical constants
c = 3e8
lambda0 = 1550e-9
f0 = c / lambda0

# FMCW parameters
B = 0.5e9          # 0.5 GHz
Tc = 100e-6        # 100 microseconds
k = B / Tc

# Sampling
fs = 20e6
N = int(Tc * fs)
t = np.arange(N) / fs

# Target parameters
R = 120.0          # meters
v = 15.0           # m/s

# Propagation delay and Doppler
tau = 2 * R / c
fd = 2 * v / lambda0

# ------------------------------------------------------------
# Baseband chirp (visualization-friendly)
# ------------------------------------------------------------
tx = np.cos(2 * np.pi * (0.5 * k * t**2))

# Delayed echo
t_echo = t - tau
rx = 0.7 * np.cos(2 * np.pi * (0.5 * k * t_echo**2 + fd * t))

# Beat signal
beat = tx * rx

# FFT
window = np.hanning(N)
spec = np.abs(fft(beat * window))[:N // 2]
freqs = fftfreq(N, 1 / fs)[:N // 2]

# ------------------------------------------------------------
# Create publication-quality figure
# ------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.linewidth": 1.2
})

fig, axes = plt.subplots(3, 1, figsize=(8, 7))

# 1. Transmitted chirp
axes[0].plot(t[:3000] * 1e6, tx[:3000], linewidth=1.2)
axes[0].set_title("Transmitted Optical FMCW Chirp")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True, linestyle='--', alpha=0.4)

# 2. Echo signal
axes[1].plot(t[:3000] * 1e6, rx[:3000], linewidth=1.2)
axes[1].set_title("Delayed Echo after Free-Space Propagation")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True, linestyle='--', alpha=0.4)

# 3. Beat spectrum
axes[2].plot(freqs / 1e6, spec, linewidth=1.2)
axes[2].set_title("Beat-Frequency Spectrum after Coherent Mixing")
axes[2].set_xlabel("Frequency (MHz)")
axes[2].set_ylabel("Magnitude")
axes[2].set_xlim(0, 6)
axes[2].grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

# Save IEEE-quality image
plt.savefig("optical_chirp_echo_graph.png",
            dpi=600,
            bbox_inches='tight',
            facecolor='white')

plt.show()
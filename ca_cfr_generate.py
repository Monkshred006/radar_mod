import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# IEEE Figure: CA-CFAR Detection Example
# ============================================================

np.random.seed(7)

# Simulated range bins
N = 256
rng = np.arange(N)

# Noise floor
noise = np.random.exponential(scale=1.0, size=N)

# Add target peaks
signal = noise.copy()
signal[60] += 8
signal[130] += 12
signal[200] += 9

# -------------------- CA-CFAR Parameters --------------------
guard = 2
train = 8
alpha = 3.5

threshold = np.zeros(N)
detections = np.zeros(N, dtype=bool)

for i in range(train + guard, N - train - guard):
    leading = signal[i - train - guard : i - guard]
    trailing = signal[i + guard + 1 : i + guard + train + 1]

    noise_est = np.mean(np.concatenate([leading, trailing]))
    threshold[i] = alpha * noise_est

    if signal[i] > threshold[i]:
        detections[i] = True

# -------------------- Plot --------------------
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif"
})

plt.figure(figsize=(8, 4.5))

plt.plot(rng, signal, linewidth=1.2, label='Range Profile')
plt.plot(rng, threshold, linewidth=1.5, linestyle='--', label='CA-CFAR Threshold')

# Mark detections
plt.scatter(rng[detections], signal[detections],
            s=50, zorder=3, label='Detected Targets')

plt.xlabel('Range Bin')
plt.ylabel('Amplitude')
plt.title('CA-CFAR Target Detection')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('cfar_detection_example.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
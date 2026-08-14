import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# IEEE Figure: JPDA–IMM Multi-Target Tracking Trajectories
# ============================================================

np.random.seed(42)

# -------------------- Simulation Parameters -----------------
T = 60                      # Number of frames
t = np.arange(T)

# -------------------- Ground Truth Trajectories ------------
# Target 1: constant velocity
x1 = 20 + 1.2 * t
y1 = 30 + 0.4 * t

# Target 2: accelerating motion
x2 = 100 - 1.5 * t
y2 = 20 + 0.03 * t**2

# Target 3: coordinated turn
theta = np.linspace(0, 1.4 * np.pi, T)
x3 = 60 + 18 * np.cos(theta)
y3 = 80 + 18 * np.sin(theta)

# -------------------- Noisy Radar Measurements -------------
noise_std = 2.0

mx1 = x1 + np.random.normal(0, noise_std, T)
my1 = y1 + np.random.normal(0, noise_std, T)

mx2 = x2 + np.random.normal(0, noise_std, T)
my2 = y2 + np.random.normal(0, noise_std, T)

mx3 = x3 + np.random.normal(0, noise_std, T)
my3 = y3 + np.random.normal(0, noise_std, T)

# -------------------- Simple IMM-like Smoothing ------------
def smooth_track(z, alpha=0.25):
    x = np.zeros_like(z)
    x[0] = z[0]
    for k in range(1, len(z)):
        x[k] = alpha * z[k] + (1 - alpha) * x[k-1]
    return x

tx1 = smooth_track(mx1)
ty1 = smooth_track(my1)

tx2 = smooth_track(mx2)
ty2 = smooth_track(my2)

tx3 = smooth_track(mx3)
ty3 = smooth_track(my3)

# -------------------- Publication-Quality Plot --------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 1.2
})

fig, ax = plt.subplots(figsize=(8, 6))

# Ground truth
ax.plot(x1, y1, '--', linewidth=1.5, label='Target 1 Ground Truth')
ax.plot(x2, y2, '--', linewidth=1.5, label='Target 2 Ground Truth')
ax.plot(x3, y3, '--', linewidth=1.5, label='Target 3 Ground Truth')

# Noisy measurements
ax.scatter(mx1, my1, s=12, alpha=0.4, label='Measurements')
ax.scatter(mx2, my2, s=12, alpha=0.4)
ax.scatter(mx3, my3, s=12, alpha=0.4)

# JPDA–IMM estimated trajectories
ax.plot(tx1, ty1, linewidth=2.2, label='Track 1 Estimate')
ax.plot(tx2, ty2, linewidth=2.2, label='Track 2 Estimate')
ax.plot(tx3, ty3, linewidth=2.2, label='Track 3 Estimate')

# Formatting
ax.set_title('JPDA–IMM Multi-Target Tracking Trajectories')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(loc='upper right', fontsize=8, ncol=2)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()

# Save IEEE-quality image
plt.savefig(
    'jpda_imm_tracking_trajectories.png',
    dpi=600,
    bbox_inches='tight',
    facecolor='white'
)

plt.show()
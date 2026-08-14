import numpy as np
import matplotlib.pyplot as plt

ranges = np.linspace(0, 150, 256)
velocities = np.linspace(-15, 15, 64)
R, V = np.meshgrid(ranges, velocities)

rd = (
    np.exp(-((R-40)**2 + (V-5)**2)/20) +
    np.exp(-((R-75)**2 + (V+8)**2)/20) +
    np.exp(-((R-120)**2 + (V-12)**2)/20) +
    0.05*np.random.rand(*R.shape)
)

plt.figure(figsize=(8, 4))
plt.imshow(rd, aspect='auto', origin='lower',
           extent=[0, 150, -15, 15], cmap='viridis')
plt.colorbar(label='Normalized Power')
plt.xlabel('Range (m)')
plt.ylabel('Velocity (m/s)')
plt.title('Simulated Range–Doppler Map')
plt.tight_layout()
plt.savefig('figures/range_doppler_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
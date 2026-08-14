import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 50)

r1 = 40 + 0.5*t
r2 = 75 - 0.8*t
r3 = 120 + 1.2*t

plt.figure(figsize=(7, 4))
plt.plot(t, r1, label='Track 1')
plt.plot(t, r2, label='Track 2')
plt.plot(t, r3, label='Track 3')
plt.xlabel('Frame')
plt.ylabel('Estimated Range (m)')
plt.title('JPDA–IMM Multi-Target Tracking')
plt.legend()
plt.tight_layout()
plt.savefig('figures/tracking_trajectories.png', dpi=300, bbox_inches='tight')
plt.close()
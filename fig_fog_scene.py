import numpy as np
import matplotlib.pyplot as plt

img = np.zeros((200, 300))
img[80:120, 120:180] = 1.0

fog = np.random.normal(0.5, 0.15, img.shape)
foggy = np.clip(0.4*img + 0.6*fog, 0, 1)

plt.figure(figsize=(6, 4))
plt.imshow(foggy, cmap='gray')
plt.axis('off')
plt.title('Synthetic Fog-Degraded Observation')
plt.tight_layout()
plt.savefig('figures/foggy_scene.png', dpi=300, bbox_inches='tight')
plt.close()
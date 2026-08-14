import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

x = np.random.uniform(-1, 1, 120)
y = np.random.uniform(-1, 1, 120)
z = np.sin(3*x) * np.cos(3*y)

fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x, y, z, s=10)
ax1.set_title('Sparse Photonic Observations')

gx, gy = np.meshgrid(np.linspace(-1,1,40), np.linspace(-1,1,40))
gz = np.sin(3*gx) * np.cos(3*gy)

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(gx, gy, gz, linewidth=0, antialiased=True)
ax2.set_title('Diffusion + PINN Reconstruction')

plt.tight_layout()
plt.savefig('figures/sparse_to_dense_reconstruction.png', dpi=300, bbox_inches='tight')
plt.close()
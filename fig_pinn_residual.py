import matplotlib.pyplot as plt

methods = ['No PINN', 'PINN Enabled']
values = [1.0, 0.352]

plt.figure(figsize=(5, 4))
plt.bar(methods, values)
plt.ylabel('Normalized Residual Energy')
plt.title('Physics Residual Reduction')
plt.tight_layout()
plt.savefig('figures/pinn_residual_reduction.png', dpi=300, bbox_inches='tight')
plt.close()
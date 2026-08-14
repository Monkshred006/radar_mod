import matplotlib.pyplot as plt

configs = ['Full', '-CFAR', '-Diffusion', '-PINN', '-IMM']
mota = [93.33, 78.89, 84.44, 87.12, 89.51]

plt.figure(figsize=(7, 4))
plt.plot(configs, mota, marker='o', linewidth=2)
plt.ylabel('MOTA (%)')
plt.title('Ablation Study of PhotonPINN-SLM')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figures/ablation_study.png', dpi=300, bbox_inches='tight')
plt.close()
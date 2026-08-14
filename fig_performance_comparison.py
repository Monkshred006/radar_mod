import matplotlib.pyplot as plt

methods = ['EKF', 'JPDA', 'JPDA+IMM']
mota = [47.78, 96.67, 100.0]

plt.figure(figsize=(6, 4))
plt.bar(methods, mota)
plt.ylabel('MOTA (%)')
plt.ylim(0, 110)
plt.title('Tracking Performance Comparison')
plt.tight_layout()
plt.savefig('figures/performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
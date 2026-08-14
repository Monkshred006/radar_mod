import subprocess
import os

os.makedirs('figures', exist_ok=True)
scripts = [
    'fig_system_architecture.py',
    'fig_range_doppler.py',
    'fig_sparse_dense.py',
    'fig_pinn_residual.py',
    'fig_tracking_trajectories.py',
    'fig_ablation_study.py',
    'fig_fog_scene.py',
    'fig_performance_comparison.py'
]

for s in scripts:
    subprocess.run(['python', s], check=True)

print('All IEEE paper figures generated successfully.')
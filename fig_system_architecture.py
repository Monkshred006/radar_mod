import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Create figure
fig, ax = plt.subplots(figsize=(18, 4))
ax.set_xlim(0, 18)
ax.set_ylim(0, 2)
ax.axis('off')

# IMPORTANT: use REAL newlines, not \\n
blocks = [
    (0.3,  "NIR Structured\nIllumination"),
    (3.3,  "Fog / Haze\nEnvironment"),
    (6.3,  "NIR Camera /\nPhotonic Sensor"),
    (9.3,  "Sparse 2D\nPoint Extraction"),
    (12.3, "Diffusion + PINN\nSLM (Uno Q)"),
    (15.3, "Dense 3D\nReconstruction")
]

box_width = 2.4
box_height = 0.9
y = 0.55

# Draw boxes
for x, label in blocks:
    rect = FancyBboxPatch(
        (x, y),
        box_width,
        box_height,
        boxstyle="round,pad=0.04",
        edgecolor="black",
        facecolor="white",
        linewidth=2
    )
    ax.add_patch(rect)

    ax.text(
        x + box_width / 2,
        y + box_height / 2,
        label,
        ha='center',
        va='center',
        fontsize=11,
        fontweight='bold'
    )

# Draw arrows
for i in range(len(blocks) - 1):
    x1 = blocks[i][0] + box_width
    x2 = blocks[i + 1][0]

    ax.annotate(
        '',
        xy=(x2, y + box_height / 2),
        xytext=(x1, y + box_height / 2),
        arrowprops=dict(arrowstyle='->', lw=2)
    )

# Save high-quality image
plt.tight_layout()
plt.savefig('fixed_system_architecture_v2.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
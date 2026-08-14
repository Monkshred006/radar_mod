import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ============================================================
# IEEE Figure: Simulation and Processing Workflow
# ============================================================

# Create figure
fig, ax = plt.subplots(figsize=(16, 4))
ax.set_xlim(0, 16)
ax.set_ylim(0, 2)
ax.axis('off')

# Workflow blocks
blocks = [
    (0.2,  "Optical Chirp\nGeneration"),
    (2.2,  "Free-Space\nPropagation"),
    (4.2,  "Coherent\nOptical Mixing"),
    (6.2,  "Photodetection\n+ ADC"),
    (8.2,  "Range–Doppler\nProcessing"),
    (10.2, "DDPM + PINN\nEnhancement"),
    (12.2, "CA-CFAR\nDetection"),
    (14.2, "JPDA–IMM\nTracking")
]

box_w = 1.5
box_h = 0.8
y = 0.6

# Draw blocks
for x, label in blocks:
    box = FancyBboxPatch(
        (x, y),
        box_w,
        box_h,
        boxstyle="round,pad=0.04",
        linewidth=2,
        edgecolor="black",
        facecolor="white"
    )
    ax.add_patch(box)

    ax.text(
        x + box_w / 2,
        y + box_h / 2,
        label,
        ha='center',
        va='center',
        fontsize=10,
        fontweight='bold'
    )

# Draw arrows
for i in range(len(blocks) - 1):
    x1 = blocks[i][0] + box_w
    x2 = blocks[i + 1][0]

    ax.annotate(
        '',
        xy=(x2, y + box_h / 2),
        xytext=(x1, y + box_h / 2),
        arrowprops=dict(arrowstyle='->', lw=2)
    )

# Title
plt.title("PhotonPINN-Radar Simulation and Processing Workflow",
          fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()

# Save high-resolution IEEE figure
plt.savefig("simulation_processing_workflow.png",
            dpi=600,
            bbox_inches='tight',
            facecolor='white')

plt.show()
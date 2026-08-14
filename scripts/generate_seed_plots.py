"""Generate seed comparison plots: V0_seed_comparison.png and V0_per_class_seed_comparison.png."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("C:/Users/worka/research/photonpinn/results/photon_v0")
BRAIN_DIR = Path("C:/Users/worka/.gemini/antigravity/brain/2df42c78-fd32-41b4-81c6-4b5d4d54f121")

seeds = ["Seed 42", "Seed 123", "Seed 456"]
macro_f1s = [0.8749, 0.8796, 0.8589]
test_accs = [0.8800, 0.8800, 0.8667]

# 1. Seed Comparison Plot
fig, ax = plt.subplots(figsize=(7, 4.5))
fig.patch.set_facecolor("#ffffff")

bars = ax.bar(seeds, macro_f1s, color=["#1f77b4", "#2ca02c", "#ff7f0e"], width=0.45, edgecolor="#333333", linewidth=1.2, zorder=3)
mean_f1 = np.mean(macro_f1s)
ax.axhline(mean_f1, color="#d62728", linestyle="--", linewidth=1.8, label=f"Mean Test Macro-F1: {mean_f1:.4f}", zorder=4)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylim([0.70, 1.0])
ax.set_ylabel("Test Macro-F1 Score", fontsize=12, fontweight="bold")
ax.set_title("PhotonV0 Seed Stability Comparison (RaDICaL Test Set)", fontsize=13, fontweight="bold", pad=12)
ax.grid(True, linestyle=":", alpha=0.6, zorder=0)
ax.legend(loc="lower right", fontsize=11)
plt.tight_layout()

p1 = OUTPUT_DIR / "V0_seed_comparison.png"
plt.savefig(p1, dpi=200)
if BRAIN_DIR.exists():
    plt.savefig(BRAIN_DIR / "V0_seed_comparison.png", dpi=200)
plt.close()
print(f"Saved {p1}")

# 2. Per-Class Seed Comparison Plot
classes = ["Empty", "Pedestrian", "Cyclist", "Vehicle"]
f1_seed42 = [0.9268, 0.8889, 0.8000, 0.8837]
f1_seed123 = [0.8571, 0.9444, 0.8387, 0.8780]
f1_seed456 = [0.9500, 0.9697, 0.6923, 0.8235]

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#ffffff")

rects1 = ax.bar(x - width, f1_seed42, width, label="Seed 42 (V0.1)", color="#1f77b4", edgecolor="#333", linewidth=1.0, zorder=3)
rects2 = ax.bar(x, f1_seed123, width, label="Seed 123 (V0.2)", color="#2ca02c", edgecolor="#333", linewidth=1.0, zorder=3)
rects3 = ax.bar(x + width, f1_seed456, width, label="Seed 456 (V0.3)", color="#ff7f0e", edgecolor="#333", linewidth=1.0, zorder=3)

# Highlight Cyclist
ax.axvspan(1.5, 2.5, color="#ffeedd", alpha=0.5, label="Cyclist Target Domain", zorder=1)

ax.set_ylabel("F1 Score", fontsize=12, fontweight="bold")
ax.set_title("PhotonV0 Per-Class Test F1 Stability Across Seeds", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
ax.set_ylim([0.5, 1.05])
ax.grid(True, linestyle=":", alpha=0.6, zorder=0)
ax.legend(loc="lower right", fontsize=10)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
p2 = OUTPUT_DIR / "V0_per_class_seed_comparison.png"
plt.savefig(p2, dpi=200)
if BRAIN_DIR.exists():
    plt.savefig(BRAIN_DIR / "V0_per_class_seed_comparison.png", dpi=200)
plt.close()
print(f"Saved {p2}")

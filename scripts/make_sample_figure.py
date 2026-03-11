"""Generate a compact 2×3 sample data figure for the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATEGORIES = ["biological", "building", "vehicle"]
IDX = "00005"

fig, axes = plt.subplots(2, 3, figsize=(7, 3.2))

for col, cat in enumerate(CATEGORIES):
    img_path = os.path.join(BASE, "dataset_lineart", cat, "train", "images", f"{IDX}.jpg")
    tgt_path = os.path.join(BASE, "dataset_lineart", cat, "train", "targets", f"{IDX}.png")

    img = Image.open(img_path).convert("RGB")
    tgt = Image.open(tgt_path).convert("L")

    axes[0, col].imshow(img)
    axes[0, col].set_title(cat.capitalize(), fontsize=10, fontweight="bold")
    axes[0, col].axis("off")

    axes[1, col].imshow(tgt, cmap="gray", vmin=0, vmax=255)
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Input Photo", fontsize=9)
axes[1, 0].set_ylabel("Line Art Target", fontsize=9)
for row in range(2):
    axes[row, 0].tick_params(left=False, labelleft=False)

plt.tight_layout(pad=0.3)
out = os.path.join(BASE, "report", "figures", "data_samples.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved to {out}")

"""2x2 enhancer/recognizer domain matrix — the fair VideoMamba comparison."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/ssd1/wfs/project/lowlight/compare/results"
# rows = recognizer training domain; cols = test-frame enhancement
rows = ["VideoMamba\ntrained on RF\n(released)", "VideoMamba\ntrained on CIDNet\n(finetuned)"]
cols = ["RF-enhanced\ntest", "CIDNet-enhanced\ntest"]
M = np.array([[0.6929, 0.5633],
              [0.6719, 0.6538]])

fig, ax = plt.subplots(figsize=(7.2, 5.2))
im = ax.imshow(M, cmap="YlGnBu", vmin=0.5, vmax=0.72)
ax.set_xticks([0, 1]); ax.set_xticklabels(cols)
ax.set_yticks([0, 1]); ax.set_yticklabels(rows)
ax.set_xlabel("test-frame enhancement (deployed enhancer)")
for i in range(2):
    for j in range(2):
        matched = i == j
        ax.text(j, i, f"{M[i,j]:.4f}", ha="center", va="center",
                fontsize=17, fontweight="bold" if matched else "normal",
                color="white" if M[i, j] > 0.64 else "black")
        if matched:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor="red", lw=3))
ax.set_title("VideoMamba top-1 on ARID split-1 (n=1768)\n"
             "red = matched recognizer↔enhancer (the FAIR comparison)", fontsize=11)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="top-1 accuracy")
fig.tight_layout()
fig.savefig(f"{R}/arid_domain_matrix.png", dpi=120)
print(f"saved {R}/arid_domain_matrix.png")

# summary bars: the story in one view
fig2, ax2 = plt.subplots(figsize=(8.5, 4.6))
labels = ["off\n(dark)", "CIDNet\n+ RF-recognizer\n(unfair, orig)", "CIDNet\n+ CIDNet-recognizer\n(fair)", "RF\n+ RF-recognizer\n(fair)"]
vals = [0.2449, 0.5633, 0.6538, 0.6929]
colors = ["#8a8f98", "#f0b27a", "#e08a3c", "#4c78c9"]
bars = ax2.bar(labels, vals, color=colors)
for b, v in zip(bars, vals):
    ax2.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
ax2.set_ylabel("VideoMamba top-1"); ax2.set_ylim(0, 0.78)
ax2.set_title("Fair action-recognition comparison (ARID split-1, n=1768)\n"
              "adapting the recognizer to CIDNet recovers +9.0 pts; residual gap to RF ≈ 3.9 pts")
ax2.grid(axis="y", alpha=0.3)
fig2.tight_layout()
fig2.savefig(f"{R}/arid_fair_comparison.png", dpi=120)
print(f"saved {R}/arid_fair_comparison.png")

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = r"C:\Users\Admin\.gemini\antigravity\scratch\quattucker_zigzag_paper\figures"
os.makedirs(OUT, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.3), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.25]})
for ax in (ax1, ax2):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

C_LAV   = "#DFE4F5"
C_CREAM = "#F7F8DC"
C_PEACH = "#FBE2C4"
C_PINK  = "#F9DCD8"
C_GREEN = "#D9EFE0"
C_INK   = "#1A1A1A"
C_RED   = "#B03A2E"

plt.rcParams["font.family"] = "sans-serif"

def box(ax, x, y, w, h, txt, fc, fw="semibold", fs=8.8):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.18",
                       facecolor=fc, edgecolor=C_INK, linewidth=1.15, zorder=3)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs, fontweight=fw, color=C_INK, zorder=4)

def arr(ax, x1, y1, x2, y2, col=C_INK, ls="-", lw=1.35):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.28",
                                color=col, lw=lw, ls=ls, shrinkA=0, shrinkB=0), zorder=4)

# (a) Standard Sequential Stack
ax1.set_title("(a) Standard Sequential Stack (1 Path)", fontsize=10.2, fontweight="bold", pad=6)
box(ax1, 0.3, 2.1, 1.7, 1.0, "Block 0\n(h0)", C_CREAM)
box(ax1, 2.8, 2.1, 1.7, 1.0, "Block 1\n(h1)", C_LAV)
box(ax1, 5.3, 2.1, 1.7, 1.0, "Block 2\n(h2)", C_PEACH)
box(ax1, 7.8, 2.1, 1.7, 1.0, "Block 3\n(h3)", C_GREEN)
arr(ax1, 2.05, 2.6, 2.75, 2.6)
arr(ax1, 4.55, 2.6, 5.25, 2.6)
arr(ax1, 7.05, 2.6, 7.75, 2.6)
ax1.text(5.0, 0.85, r"Single Jacobian Path:  $\partial h_3 / \partial h_0 = J_3 J_2 J_1$",
         ha="center", va="center", fontsize=8.8, fontweight="semibold", color="#333333")

# (b) Braided ZigZag Diamond
ax2.set_title("(b) Braided ZigZag Diamond (3 Concurrent Paths, 0 Extra FLOPs)", fontsize=10.2, fontweight="bold", pad=6)
box(ax2, 0.3, 2.1, 1.7, 1.0, "Block 0\n(h0)", C_CREAM)
box(ax2, 3.8, 3.6, 2.2, 1.0, "Block 2\n(Coarse h2)", C_PEACH, fw="bold")
box(ax2, 3.8, 0.6, 2.2, 1.0, "Block 1\n(Refine h1)", C_LAV, fw="bold")
box(ax2, 7.8, 2.1, 1.8, 1.0, "Block 3\n(Fuse h3)", C_GREEN, fw="bold")

arr(ax2, 2.05, 2.85, 3.75, 4.10)
arr(ax2, 2.05, 2.35, 3.75, 1.10)
arr(ax2, 4.90, 3.55, 4.90, 1.65, col=C_RED, ls="--", lw=1.8)
ax2.text(5.15, 2.60, "Top-Down\nFeedback\n0.5(h0+h2)", ha="left", va="center",
         fontsize=7.8, fontweight="bold", color=C_RED)
arr(ax2, 6.05, 4.10, 7.75, 2.85)
arr(ax2, 6.05, 1.10, 7.75, 2.35)
ax2.text(8.70, 0.75, r"$\frac{1}{2}J_3 J_2 + \frac{1}{2}J_3 J_1 + \frac{1}{4}J_3 J_1 J_2$",
         ha="center", va="center", fontsize=8.5, fontweight="semibold", color="#333333")

fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "fig7_zigzag_routing.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig7_zigzag_routing.pdf"), bbox_inches="tight")
plt.close(fig)
print("SUCCESS: Generated fig7_zigzag_routing.png and .pdf")

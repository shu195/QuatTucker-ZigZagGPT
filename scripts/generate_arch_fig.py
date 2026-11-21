import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Polygon, Circle

OUT = r"C:\Users\Admin\.gemini\antigravity\scratch\quattucker_zigzag_paper\figures"
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(10.5, 9.2), dpi=300)
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 9.2)
ax.axis("off")

# Exact Qwen3-Next pastel palette
C_LAV   = "#DFE4F5"  # Linear / QuatLinear / FFN
C_CREAM = "#F7F8DC"  # Attention / Tucker Core
C_PEACH = "#FBE2C4"  # LayerNorm
C_PINK  = "#F9DCD8"  # Braided Fusion / Reshape
C_YEL   = "#F9EA9C"  # SiLU circle
C_DASH  = "#7A7A7A"  # Dashed border
C_INK   = "#1A1A1A"  # Crisp lines & text

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

def rbox(x, y, w, h, text, fc, ec=C_INK, lw=1.15, fs=9.5, fw="semibold", ls="-", shadow=False):
    if shadow:
        sh = FancyBboxPatch((x + 0.04, y - 0.04), w, h, boxstyle="round,pad=0.04,rounding_size=0.14",
                            facecolor="#00000018", edgecolor="none", zorder=2)
        ax.add_patch(sh)
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.14",
                       facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3)
    ax.add_patch(p)
    if text:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, fontweight=fw, color=C_INK, zorder=4)

def dash_panel(x, y, w, h):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.25",
                       facecolor="#FAFAFA", edgecolor=C_DASH, linewidth=1.15, linestyle="--", zorder=1)
    ax.add_patch(p)

def trap(x, y, w, h, text, fc=C_LAV, invert=False, fs=8.8):
    dx = 0.16
    if not invert:
        pts = [(x, y), (x + w, y), (x + w - dx, y + h), (x + dx, y + h)]
    else:
        pts = [(x + dx, y), (x + w - dx, y), (x + w, y + h), (x, y + h)]
    poly = Polygon(pts, closed=True, facecolor=fc, edgecolor=C_INK, linewidth=1.15, zorder=3)
    ax.add_patch(poly)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, fontweight="semibold", color=C_INK, zorder=4)

def circle_op(cx, cy, sym, fc="white", r=0.15, fs=10, ec=C_INK):
    c = Circle((cx, cy), r, facecolor=fc, edgecolor=ec, linewidth=1.15, zorder=5)
    ax.add_patch(c)
    ax.text(cx, cy, sym, ha="center", va="center", fontsize=fs, fontweight="bold", color=C_INK, zorder=6)

def arrow(x1, y1, x2, y2, color=C_INK, ls="-", lw=1.15):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.28",
                                color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0), zorder=4)

def line(xs, ys, color=C_INK, ls="-", lw=1.15):
    ax.plot(xs, ys, color=color, ls=ls, lw=lw, zorder=2)

# ==============================================================================
# LEFT COLUMN: 2x Braided ZigZag Diamond Macro-Architecture
# ==============================================================================
dash_panel(0.65, 0.45, 3.25, 8.35)
ax.text(0.32, 4.62, "2×", ha="center", va="center", fontsize=13, fontweight="bold", color=C_INK)

# Bottom input arrow
arrow(2.275, 0.12, 2.275, 0.85)
ax.text(2.42, 0.16, "Token + Pos Emb (d=384)", ha="left", va="center", fontsize=8.3, fontweight="semibold", color="#333333")

# 4 Blocks inside the ZigZag Diamond (bottom to top: B0 -> B2 -> B1 -> B3)
# Block 0
rbox(1.15, 0.88, 2.25, 0.50, "LayerNorm", C_PEACH, fs=9)
rbox(1.15, 1.56, 2.25, 0.62, "Block 0  (Attn + QuatFFN)", C_CREAM, fs=9, fw="bold")
arrow(2.275, 1.38, 2.275, 1.56)
circle_op(2.275, 2.42, "+", r=0.13, fs=10)
arrow(2.275, 2.18, 2.275, 2.29)
# Residual around Block 0
line([2.275, 0.88, 0.88], [0.72, 0.72, 2.42])
arrow(0.88, 2.42, 2.14, 2.42)
ax.text(2.58, 2.58, "h0", ha="left", va="center", fontsize=8.5, fontweight="bold", color="#333333")

# Block 2 (Coarse Jump)
arrow(2.275, 2.55, 2.275, 2.86)
rbox(1.15, 2.86, 2.25, 0.50, "LayerNorm", C_PEACH, fs=9)
rbox(1.15, 3.54, 2.25, 0.62, "Block 2  (Coarse Jump)", C_CREAM, fs=9, fw="bold")
arrow(2.275, 3.36, 2.275, 3.54)
circle_op(2.275, 4.40, "+", r=0.13, fs=10)
arrow(2.275, 4.16, 2.275, 4.27)
line([2.275, 0.88, 0.88], [2.68, 2.68, 4.40])
arrow(0.88, 4.40, 2.14, 4.40)
ax.text(2.58, 4.56, "h2", ha="left", va="center", fontsize=8.5, fontweight="bold", color="#333333")

# Braided Merge 1: 0.5(h0 + h2) -> Block 1 (Top-Down Refine)
arrow(2.275, 4.53, 2.275, 4.86)
rbox(1.15, 4.86, 2.25, 0.44, "Braided Mix: 0.5 (h0 + h2)", C_PINK, fs=8.5)
# Skip h0 from right side into Braided Mix 1
line([2.40, 3.68, 3.68], [2.42, 2.42, 5.08], color="#B03A2E", lw=1.25)
arrow(3.68, 5.08, 3.42, 5.08, color="#B03A2E", lw=1.25)

arrow(2.275, 5.30, 2.275, 5.56)
rbox(1.15, 5.56, 2.25, 0.62, "Block 1  (Top-Down Refine)", C_LAV, fs=9, fw="bold")
circle_op(2.275, 6.42, "+", r=0.13, fs=10)
arrow(2.275, 6.18, 2.275, 6.29)
line([2.275, 0.88, 0.88], [5.43, 5.43, 6.42])
arrow(0.88, 6.42, 2.14, 6.42)
ax.text(2.58, 6.42, "h1", ha="left", va="center", fontsize=8.5, fontweight="bold", color="#333333")

# Braided Merge 2: 0.5(h1 + h2) -> Block 3 (Fuse)
arrow(2.275, 6.55, 2.275, 6.86)
rbox(1.15, 6.86, 2.25, 0.44, "Braided Mix: 0.5 (h1 + h2)", C_PINK, fs=8.5)
# Skip h2 from right side into Braided Mix 2
line([2.40, 3.56, 3.56], [4.40, 4.40, 7.08], color="#B03A2E", lw=1.25)
arrow(3.56, 7.08, 3.42, 7.08, color="#B03A2E", lw=1.25)

arrow(2.275, 7.30, 2.275, 7.56)
rbox(1.15, 7.56, 2.25, 0.62, "Block 3  (Final Fuse)", C_LAV, fs=9, fw="bold")
circle_op(2.275, 8.42, "+", r=0.13, fs=10)
arrow(2.275, 8.18, 2.275, 8.29)
line([2.275, 0.88, 0.88], [7.43, 7.43, 8.42])
arrow(0.88, 8.42, 2.14, 8.42)

# Top output arrow
arrow(2.275, 8.55, 2.275, 9.05)

# ==============================================================================
# TOP-RIGHT PANEL: Shared 4D Quaternion + Tucker SwiGLU FFN
# ==============================================================================
dash_panel(4.45, 4.90, 5.75, 3.90)
# Legend badges top-right
circle_op(8.85, 8.50, "σ", fc=C_YEL, r=0.15, fs=9.5, ec="#B7950B")
ax.text(9.08, 8.50, "SiLU", ha="left", va="center", fontsize=9, fontweight="semibold", color=C_INK)
circle_op(8.85, 8.10, "H", fc="#FADBD8", r=0.15, fs=8.8, ec="#922B21")
ax.text(9.08, 8.10, "4D Quat", ha="left", va="center", fontsize=8.6, fontweight="semibold", color="#78281F")
ax.text(4.65, 8.55, "QuatTucker SwiGLU\n(4× Compressed)", ha="left", va="top",
        fontsize=8.8, fontweight="bold", color="#333333")

# Input from bottom of panel
line([7.32, 7.32], [4.80, 5.20])
line([5.95, 8.70], [5.20, 5.20])
arrow(5.95, 5.20, 5.95, 5.45)
arrow(8.70, 5.20, 8.70, 5.45)

# Two Quaternion Trapezoids: U_gate and U_val
trap(5.05, 5.45, 1.80, 0.50, "QuatLinear U_gate", fc=C_LAV, invert=True, fs=8.3)
trap(7.80, 5.45, 1.80, 0.50, "QuatLinear U_val",  fc=C_LAV, invert=True, fs=8.3)

# SiLU on U_gate branch
arrow(5.95, 5.95, 5.95, 6.20)
circle_op(5.95, 6.35, "σ", fc=C_YEL, r=0.15, fs=9.5, ec="#B7950B")
line([5.95, 5.95, 7.18], [6.50, 6.80, 6.80])
arrow(7.05, 6.80, 7.18, 6.80)

# U_val branch to multiply
line([8.70, 8.70, 7.46], [5.95, 6.80, 6.80])
arrow(7.58, 6.80, 7.46, 6.80)

# Multiply node
circle_op(7.32, 6.80, "×", r=0.14, fs=10)
arrow(7.32, 6.94, 7.32, 7.20)

# Block-Diagonal Tucker Core G[l]
rbox(5.32, 7.20, 4.00, 0.56, "Block-Diagonal Tucker Core  G[l]  (64×64)", C_CREAM, fs=9.2, fw="bold", shadow=True)
arrow(7.32, 7.76, 7.32, 8.00)

# Top Quaternion Output Trapezoid
trap(6.32, 8.00, 2.00, 0.46, "QuatLinear V_out (H)", fc=C_LAV, invert=False, fs=8.3)
arrow(7.32, 8.46, 7.32, 9.00)

# ==============================================================================
# BOTTOM-RIGHT PANEL: Independent Dense Causal FlashAttention
# ==============================================================================
dash_panel(4.45, 0.45, 5.75, 4.10)
ax.text(4.65, 4.35, "Dense Per-Layer\nFlashAttention", ha="left", va="top",
        fontsize=8.8, fontweight="bold", color="#333333")

# Input splits into Q, K, V
line([7.32, 7.32], [0.35, 0.85])
line([5.55, 9.10], [0.85, 0.85])
arrow(5.55, 0.85, 5.55, 1.15)
arrow(7.32, 0.85, 7.32, 1.15)
arrow(9.10, 0.85, 9.10, 1.15)

trap(4.85, 1.15, 1.40, 0.48, "Linear Q", fc=C_LAV, invert=True, fs=8.5)
trap(6.62, 1.15, 1.40, 0.48, "Linear K", fc=C_LAV, invert=True, fs=8.5)
trap(8.40, 1.15, 1.40, 0.48, "Linear V", fc=C_LAV, invert=True, fs=8.5)

arrow(5.55, 1.63, 5.55, 2.38)
arrow(7.32, 1.63, 7.32, 2.38)
arrow(9.10, 1.63, 9.10, 2.38)
ax.text(5.35, 2.02, "q", fontsize=11, style="italic", fontweight="bold")
ax.text(7.12, 2.02, "k", fontsize=11, style="italic", fontweight="bold")
ax.text(8.90, 2.02, "v", fontsize=11, style="italic", fontweight="bold")

# Scaled Dot-Product FlashAttention box
rbox(5.05, 2.40, 4.55, 0.62, "Scaled Dot-Product FlashAttention (Causal)", C_CREAM, fs=9.5, fw="bold", shadow=True)
arrow(7.32, 3.02, 7.32, 3.38)

# Output projection trapezoid
trap(6.42, 3.38, 1.80, 0.48, "Linear Proj", fc=C_LAV, invert=False, fs=8.8)
arrow(7.32, 3.86, 7.32, 4.70)

# ==============================================================================
# DASHED CALLOUT ARROWS FROM LEFT BLOCKS TO RIGHT ZOOM-IN PANELS
# ==============================================================================
line([3.40, 4.12, 4.12], [5.87, 5.87, 6.85], color=C_INK, ls="--", lw=1.15)
arrow(4.12, 6.85, 4.42, 6.85, color=C_INK, ls="--", lw=1.15)

line([3.40, 4.12, 4.12], [1.87, 1.87, 2.50], color=C_INK, ls="--", lw=1.15)
arrow(4.12, 2.50, 4.42, 2.50, color=C_INK, ls="--", lw=1.15)

fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(OUT, "fig5_architecture_schematic_v2.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig5_architecture_schematic_v2.pdf"), bbox_inches="tight")
plt.close(fig)
print("SUCCESS: Generated fig5_architecture_schematic_v2.png and .pdf")

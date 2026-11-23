import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Publication Styling (NeurIPS / ICML) ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

BASE_DIR = r"C:\Users\Admin\.gemini\antigravity\scratch\quattucker_zigzag_paper"
FIG_DIR = os.path.join(BASE_DIR, "figures")
with open(os.path.join(BASE_DIR, "data", "gpt2_scale_a100.json"), "r") as f:
    data = json.load(f)

C_BLUE   = "#1f4e79"
C_TEAL   = "#0e8a8a"
C_ORANGE = "#d95f02"
C_PURPLE = "#6a3d9a"
C_RED    = "#c0392b"
C_GRAY   = "#555555"

def save_fig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"))
    plt.close(fig)
    print(f"Saved {name}.pdf/.png")

# ==============================================================================
# FIG 9: FULL 0 -> 2.55B CONTINUAL 2D SCALING TRAJECTORY (HERO FIGURE)
# ==============================================================================
s2 = data["stage2_full_1B"]
s3 = data["stage3_grown_1_55B"]
s4 = data["stage4_depth16_2_55B"]
gpt2_val = s2["openai_gpt2_small_ref"]["val_loss"]

fig, ax = plt.subplots(figsize=(9.2, 4.4))

# Stage 2 (0 -> 1.05B)
t2 = np.array(s2["tokens_M"]) / 1000.0
v2 = np.array(s2["val_loss"])
ax.plot(t2, v2, color=C_BLUE, lw=2.2, label="Stage 2: L=12, rank=6144 (77.6M / 39.0M bb)")

# Stage 3 (1.05B -> 1.55B)
t3 = np.array([1048.7] + s3["tokens_M"]) / 1000.0
v3 = np.array([s3["step0_val_loss"]] + s3["val_loss"])
ax.plot(t3, v3, color=C_ORANGE, lw=2.2, label="Stage 3: Width-Grown rank=8192 (81.1M / 42.5M bb)")

# Stage 4 (1.55B -> 2.55B)
s4_steps = np.array(s4["milestones"]["steps"])
t4 = (1548.7 + s4_steps * 0.098304) / 1000.0
v4 = np.array(s4["milestones"]["val_loss"])
ax.plot(t4, v4, color=C_TEAL, lw=2.5, label="Stage 4: Depth-Grown L=16 (95.3M / 56.7M bb)")

# Shaded stage backgrounds
ax.axvspan(0.0, 1.0487, color=C_BLUE, alpha=0.04)
ax.axvspan(1.0487, 1.5487, color=C_ORANGE, alpha=0.05)
ax.axvspan(1.5487, 2.56, color=C_TEAL, alpha=0.05)

# Vertical growth lines
ax.axvline(1.0487, color=C_ORANGE, ls=":", lw=1.4)
ax.axvline(1.5487, color=C_TEAL, ls=":", lw=1.4)

ax.annotate("Width Growth\n6144 → 8192\n(+3.5M, Δ=0.0067)",
            xy=(1.0487, 3.62), xytext=(0.72, 4.35),
            arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.2),
            fontsize=8.5, fontweight="bold", color=C_ORANGE,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_ORANGE, alpha=0.9))

ax.annotate("Depth Growth\nL=12 → 16 (4 Diamonds)\n(+14.2M, Δ=0.000000)",
            xy=(1.5487, 3.567), xytext=(1.58, 4.35),
            arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.2),
            fontsize=8.5, fontweight="bold", color=C_TEAL,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_TEAL, alpha=0.9))

# Final point callout
ax.scatter([t4[-1]], [v4[-1]], color=C_TEAL, s=55, zorder=5)
ax.annotate("Best: 3.4647 (31.97 PPL)\n41/41 monotonic drops",
            xy=(t4[-1], v4[-1]), xytext=(1.88, 3.78),
            arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.2),
            fontsize=8.5, fontweight="bold", color=C_TEAL,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_TEAL, alpha=0.95))

# OpenAI GPT-2 reference
ax.axhline(gpt2_val, color=C_RED, ls="--", lw=1.5, label=f"OpenAI gpt2 ref (124.4M / 85.0M bb, ~10B tok): {gpt2_val:.4f}")

ax.set_xlim(0.0, 2.60)
ax.set_ylim(3.25, 5.55)
ax.set_xlabel("Cumulative Pre-Training Tokens (Billions)")
ax.set_ylabel("Held-Out FineWeb-Edu Val Loss (nats)")
ax.set_title("QuatTucker-ZigZagGPT: Continuous 2D Zero-Loss Scaling Trajectory (0 → 2.55B Tokens)")
ax.legend(loc="upper right", frameon=True, framealpha=0.92)
save_fig(fig, "fig9_continual_2d_scaling")

# ==============================================================================
# FIG 10: ZERO-LOSS 2D GROWTH MECHANICS & PARAMETER EFFICIENCY
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.3), gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.28})

# Panel A: Schematic of 2D Zero-Loss Growth
ax1.axis("off")
ax1.set_xlim(0, 10.2); ax1.set_ylim(0, 6.2)
ax1.set_title("A. 2D Zero-Loss Continual Growth Operators", pad=8)

# Box 1: Base L=12, rank=6144
b1 = patches.FancyBboxPatch((0.2, 3.6), 2.8, 2.0, boxstyle="round,pad=0.15", fc="#eaf2f8", ec=C_BLUE, lw=1.6)
ax1.add_patch(b1)
ax1.text(1.6, 5.15, "Stage 2 Base", ha="center", fontweight="bold", color=C_BLUE, fontsize=8.8)
ax1.text(1.6, 4.3, "3 Diamonds (L=12)\nrank = 6144\n39.0M bb | val 3.585", ha="center", fontsize=7.6)

# Box 2: Width-Grown L=12, rank=8192
b2 = patches.FancyBboxPatch((3.7, 3.6), 2.8, 2.0, boxstyle="round,pad=0.15", fc="#fef5e7", ec=C_ORANGE, lw=1.6)
ax1.add_patch(b2)
ax1.text(5.1, 5.15, "Stage 3 Width+", ha="center", fontweight="bold", color=C_ORANGE, fontsize=8.8)
ax1.text(5.1, 4.3, "3 Diamonds (L=12)\nrank = 8192 (+2048)\n42.5M bb | Δ=0.0067", ha="center", fontsize=7.6)

# Box 3: Depth-Grown L=16, rank=8192
b3 = patches.FancyBboxPatch((7.2, 3.6), 2.8, 2.0, boxstyle="round,pad=0.15", fc="#e8f8f5", ec=C_TEAL, lw=1.6)
ax1.add_patch(b3)
ax1.text(8.6, 5.15, "Stage 4 Depth+", ha="center", fontweight="bold", color=C_TEAL, fontsize=8.8)
ax1.text(8.6, 4.3, "4 Diamonds (L=16)\nrank = 8192\n56.7M bb | Δ=0.0000", ha="center", fontsize=7.6)

# Arrows
ax1.annotate("", xy=(3.62, 4.6), xytext=(3.08, 4.6), arrowprops=dict(arrowstyle="->", lw=1.8, color=C_ORANGE))
ax1.annotate("", xy=(7.12, 4.6), xytext=(6.58, 4.6), arrowprops=dict(arrowstyle="->", lw=1.8, color=C_TEAL))

# Exact preservation equations box
eq_box = patches.FancyBboxPatch((0.2, 0.35), 9.8, 2.65, boxstyle="round,pad=0.15", fc="#f8f9f9", ec="#aaaaaa", lw=1.1)
ax1.add_patch(eq_box)
ax1.text(5.1, 2.6, "Exact Function-Preserving Expansion Conditions", ha="center", fontweight="bold", fontsize=8.8)
ax1.text(0.45, 1.95, "• Axis 1 (Width): Δrank = 256k (64-dim Tucker blocks per 4D Quat quadrant)", fontsize=7.5)
ax1.text(0.45, 1.35, "   Init new V_out = 0 ⇒ +3.5M params (10.8× cheaper than StdGPT +37.7M)", fontsize=7.5, color=C_ORANGE)
ax1.text(0.45, 0.72, "• Axis 2 (Depth): Append Diamond d+1 with proj=0, V_out=0 ⇒ max_diff = 0", fontsize=7.5, color=C_TEAL)

# Panel B: Parameter Efficiency Comparison
labels = ["StdGPT-12\n(Baseline)", "QuatTucker-12\n(Stage 2)", "QuatTucker-12G\n(Stage 3)", "QuatTucker-16\n(Stage 4)"]
bb_vals = [85.0, 39.0, 42.5, 56.71]
colors = [C_GRAY, C_BLUE, C_ORANGE, C_TEAL]
x = np.arange(len(labels))
bars = ax2.bar(x, bb_vals, color=colors, width=0.56, edgecolor="black", linewidth=0.7)
for b, val in zip(bars, bb_vals):
    ax2.text(b.get_x() + b.get_width()/2, val + 1.8, f"{val:.1f}M", ha="center", fontweight="bold", fontsize=9)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8.0)
ax2.set_ylabel("Backbone Parameters (Millions)")
ax2.set_ylim(0, 98)
ax2.set_title("B. Backbone Size (33% Smaller at L=16)")
ax2.axhline(85.0, color=C_GRAY, ls="--", lw=1.0, alpha=0.6)
save_fig(fig, "fig10_2d_growth_mechanics")

# ==============================================================================
# FIG 11: STAGE 4 HIGH-RES MONOTONIC ZOOM & POST-TRAINING SFT/DISTILL CURVES
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))

# Panel A: Stage 4 Zoom
ax1.plot(t4, v4, color=C_TEAL, lw=2.2, marker="o", markersize=2.8, label="Depth-16 Held-Out Val")
ax1.axhline(s4["step0_val_loss"], color=C_GRAY, ls=":", lw=1.2, label=f"Step-0 Post-Grow ({s4['step0_val_loss']:.4f})")
ax1.scatter([t4[-1]], [v4[-1]], color=C_RED, s=50, zorder=5)
ax1.annotate(f"Final: {v4[-1]:.4f}\n({s4['best_ppl']:.2f} PPL)",
             xy=(t4[-1], v4[-1]), xytext=(2.18, 3.535),
             arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2),
             fontsize=8.5, fontweight="bold", color=C_RED)
ax1.set_xlabel("Cumulative Tokens (Billions)")
ax1.set_ylabel("FineWeb-Edu Val Loss (nats)")
ax1.set_title("A. Stage 4 Depth-16 Zoom (41/41 Monotonic Drops)")
ax1.legend(loc="upper right", fontsize=8)

# Panel B: Post-Training Val Curves (Qwen3.5 10k Distill vs smoltalk 100k SFT)
q_d = data["stage3_distill_qwen35_10k"]
s_d = data["stage4_sft_smoltalk_100k"]
ax2.plot(np.linspace(0, 100, len(q_d["val_loss"])), q_d["val_loss"],
         color=C_PURPLE, lw=2.2, marker="s", markersize=4.5,
         label=f"Stage 3 (L=12) + Qwen3.5-0.8B 10k Distill ({q_d['val_loss'][-1]:.4f})")
ax2.plot(np.linspace(0, 100, len(s_d["val_loss"])), s_d["val_loss"],
         color=C_TEAL, lw=2.2, marker="o", markersize=4.5,
         label=f"Stage 4 (L=16) + smoltalk 100k SFT ({s_d['val_loss'][-1]:.4f})")
ax2.set_xlabel("Post-Training Progress (%)")
ax2.set_ylabel("Held-Out Assistant Val Loss (nats)")
ax2.set_title("B. Post-Training Validation Convergence")
ax2.legend(loc="upper right", fontsize=8)
save_fig(fig, "fig11_stage4_zoom_and_sft")

# ==============================================================================
# FIG 12: PURE ROUTING ABLATION (56 SEEDS) & BACKBONE PARETO FRONTIER
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.1), gridspec_kw={"wspace": 0.28})

# Panel A: 56-Seed Pure Routing Topology Ablation
tasks = ["6-Bit Parity\nSolve Rate (24s)", "8-Bit Parity\nUnseen Test (20s)", "3D Cuboid 4-Corner\nParity (12s)"]
seq_vals  = [33.3, 75.1, 74.7]
zz_vals   = [66.7, 75.4, 82.4]
loop_vals = [79.2, 83.7, 88.5]

x = np.arange(len(tasks))
w = 0.25
ax1.bar(x - w, seq_vals,  w, label="Sequential", color="#95a5a6", edgecolor="black", lw=0.6)
ax1.bar(x,     zz_vals,   w, label="Braided ZigZag (0→2→1→3)", color=C_BLUE, edgecolor="black", lw=0.6)
ax1.bar(x + w, loop_vals, w, label="Braided + 1 Loop (L3→L1)", color=C_TEAL, edgecolor="black", lw=0.6)
for i in range(len(tasks)):
    ax1.text(x[i] + w, loop_vals[i] + 1.8, f"{loop_vals[i]:.1f}%", ha="center", fontsize=8, fontweight="bold", color=C_TEAL)
ax1.set_xticks(x)
ax1.set_xticklabels(tasks, fontsize=8.3)
ax1.set_ylabel("Accuracy / Solve Rate (%)")
ax1.set_ylim(0, 102)
ax1.set_title("A. Pure Routing Ablation (56 Seeds, Iso-Param)")
ax1.legend(loc="lower right", fontsize=7.8)

# Panel B: GPT-2 Scale Backbone Efficiency Pareto
models_p = [
    ("StandardGPT-12\n(16.4M tok)", 85.0, 230.67, C_GRAY, "s"),
    ("QuatTucker-12\n(16.4M tok)", 39.0, 215.00, C_BLUE, "o"),
    ("QuatTucker-12\n(1.0B tok)", 39.0, 36.05, C_BLUE, "D"),
    ("QuatTucker-12G\n(1.55B tok)", 42.5, 36.46, C_ORANGE, "D"),
    ("QuatTucker-16\n(2.55B tok)", 56.71, 31.97, C_TEAL, "*"),
    ("OpenAI gpt2\n(~10B tok)", 85.0, 28.33, C_RED, "^"),
]
# Plot the 1B+ scale Pareto points cleanly
pareto_pts = models_p[2:]
for lbl, bb, ppl, col, mk in pareto_pts:
    sz = 140 if mk == "*" else 85
    ax2.scatter([bb], [ppl], color=col, marker=mk, s=sz, zorder=5, edgecolor="black", lw=0.6)
    dx, dy = (2.2, 0.4) if "16" not in lbl else (-18.0, -2.5)
    if "12G" in lbl: dx, dy = (2.0, 0.8)
    if "gpt2" in lbl: dx, dy = (-20.0, 0.9)
    if "1.0B" in lbl: dx, dy = (2.0, -1.8)
    ax2.annotate(f"{lbl}\n({ppl:.2f} PPL)", xy=(bb, ppl), xytext=(bb + dx, ppl + dy),
                 fontsize=7.8, fontweight="bold" if "16" in lbl else "normal", color=col)

ax2.plot([39.0, 56.71, 85.0], [36.05, 31.97, 28.33], color="#888888", ls="--", lw=1.0, alpha=0.6)
ax2.set_xlim(32, 95)
ax2.set_ylim(25, 40)
ax2.set_xlabel("Backbone Parameters (Millions)")
ax2.set_ylabel("Held-Out FineWeb-Edu Perplexity (↓ better)")
ax2.set_title("B. Backbone Pareto Frontier (GPT-2 Scale)")
save_fig(fig, "fig12_routing_ablation_and_pareto")

print("ALL V2 FIGURES GENERATED SUCCESSFULLY!")

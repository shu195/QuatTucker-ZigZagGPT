import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

BASE = r"C:\Users\Admin\.gemini\antigravity\scratch\quattucker_zigzag_paper"
with open(os.path.join(BASE, "data", "experiments.json"), "r", encoding="utf-8") as f:
    D = json.load(f)

OUT = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
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
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "std": "#4C72B0",
    "cp": "#DD8452",
    "tuck": "#55A868",
    "quat": "#C44E52",
    "zz": "#8172B3",
    "gray": "#7F7F7F",
}

def save_fig(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    plt.close(fig)

# ------------------------------------------------------------------
# FIG 1 (FIXED): 8-Bit Parity Leaderboard + Lambda Sweep
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.5), gridspec_kw={"width_ratios": [1.5, 1.0]})

items = D["parity_8bit_5seed"][::-1]
names = [f"{x['name']} ({x['params']:,}p)" for x in items]
means = [x["mean"] for x in items]
stds  = [x["std"] for x in items]
bar_cols = [COLORS["quat"] if "CP-Cuboid" in x["name"] else
            COLORS["cp"] if "GatedBounce" in x["name"] else
            COLORS["zz"] if "ZigZag" in x["name"] else
            COLORS["std"] if "ResNet" in x["name"] else COLORS["gray"] for x in items]

y_pos = np.arange(len(names))
ax1.barh(y_pos, means, xerr=stds, color=bar_cols, alpha=0.9, capsize=3, error_kw={"lw": 1.1, "ecolor": "#222222"})
ax1.set_yticks(y_pos)
ax1.set_yticklabels(names)
ax1.set_xlabel("Unseen Test Accuracy (%) — 5-Seed Mean ± Std")
ax1.set_title("(a) 8-Bit Parity Architecture Leaderboard")
ax1.set_xlim(0, 124)
for i, (m, s) in enumerate(zip(means, stds)):
    ax1.text(m + s + 1.8, i, f"{m:.1f}%", va="center", fontsize=8, fontweight="bold")

lams = [str(x["lam"]) for x in D["lambda_sweep"]]
l_means = [x["mean"] for x in D["lambda_sweep"]]
l_stds  = [x["std"] for x in D["lambda_sweep"]]
l_br    = [x["bounce"] for x in D["lambda_sweep"]]
x_idx = np.arange(len(lams))

ax2_r = ax2.twinx()
ax2_r.spines["top"].set_visible(False)
ax2_r.spines["right"].set_visible(True)
ax2_r.grid(False)

l1 = ax2.errorbar(x_idx, l_means, yerr=l_stds, fmt="o-", color=COLORS["quat"], lw=2, capsize=3, label="Test Acc (%)")
l2 = ax2_r.plot(x_idx, l_br, "s--", color=COLORS["std"], lw=1.8, label="Inf. Bounce Rate (%)")
ax2.set_xticks(x_idx)
ax2.set_xticklabels(lams)
ax2.set_xlabel(r"Sparsity Tax ($\lambda$)")
ax2.set_ylabel("Test Accuracy (%)", color=COLORS["quat"])
ax2_r.set_ylabel("Inference Bounce Rate (%)", color=COLORS["std"])
ax2.set_title(r"(b) Annealed Scaffolding ($\lambda$ Phase Transition)")
ax2.set_ylim(35, 114)
ax2_r.set_ylim(-6, 75)

lines = [l1] + l2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc="upper center", frameon=True, framealpha=0.95)

fig.tight_layout()
save_fig(fig, "fig1_parity_and_scaffolding")

# ------------------------------------------------------------------
# FIG 2 (FIXED): Mod-53 Algorithmic Grokking Trajectories
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 4.1))
mg = D["mod53_grokking_8layer"]
steps = mg["steps"]
ax.plot(steps, mg["ResNet-8"]["curve"], "o--", color=COLORS["std"], lw=1.8, label="ResNet-8 (144,373 params)")
ax.plot(steps, mg["ZigZag-8"]["curve"], "s-", color=COLORS["zz"], lw=2.0, label="ZigZag-8 (144,373 params)")
ax.plot(steps, mg["CP-Cuboid-8"]["curve"], "^-", color=COLORS["quat"], lw=2.4, label="CP-Cuboid-8 (18,997 params — 7.6x smaller)")
ax.axhline(95.0, color="#333333", ls=":", lw=1.0, alpha=0.7)
ax.text(165, 96.8, "95% Grokking Threshold", fontsize=8, color="#333333")
ax.set_xlabel("Training Step")
ax.set_ylabel("Unseen Test Accuracy (%)")
ax.set_title("Mod-53 Addition Grokking: 7.6x Compression with Zero Slingshot")
ax.set_ylim(-3, 106)
ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.88), frameon=True, framealpha=0.95)
fig.tight_layout()
save_fig(fig, "fig2_mod53_grokking")

# ------------------------------------------------------------------
# FIG 5 (NEW): Architecture Schematic (Sequential vs ZigZag vs QuatTucker)
# ------------------------------------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 3.8), gridspec_kw={"width_ratios": [1.0, 1.35]})
for ax in (axA, axB):
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

def draw_box(ax, x, y, w, h, text, fc, ec="#222222", fw="bold", fs=8.5):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.18",
                                 facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, fontweight=fw, color="#111111")

# Panel A: Braided ZigZag Diamond (0 -> 2 -> 1 -> 3)
axA.set_title("(a) Braided ZigZag Diamond (Zero Extra FLOPs)", fontsize=10.5, fontweight="bold")
draw_box(axA, 0.3, 2.5, 1.4, 1.0, "Input\nx", "#E8ECF1")
draw_box(axA, 2.3, 2.5, 1.5, 1.0, "Block 0\n(h0)", "#D4E4F7")
draw_box(axA, 4.7, 4.2, 1.8, 1.0, "Block 2\n(Coarse h2)", "#FDE2CC")
draw_box(axA, 4.7, 0.8, 1.8, 1.0, "Block 1\n(Refine h1)", "#E2D9F3")
draw_box(axA, 7.6, 2.5, 1.8, 1.0, "Block 3\n(Fuse h3)", "#D4EDDA")

arrow = dict(arrowstyle="->,head_width=0.25,head_length=0.25", lw=1.5, color="#333333")
axA.annotate("", xy=(2.25, 3.0), xytext=(1.78, 3.0), arrowprops=arrow)
axA.annotate("", xy=(4.65, 4.7), xytext=(3.85, 3.2), arrowprops=arrow)
axA.annotate("", xy=(4.65, 1.3), xytext=(3.85, 2.8), arrowprops=arrow)
axA.annotate("", xy=(5.6, 1.9), xytext=(5.6, 4.15), arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.25", lw=1.8, color=COLORS["quat"], ls="--"))
axA.text(5.85, 3.0, "Top-Down\nFeedback", fontsize=7.5, color=COLORS["quat"], fontweight="bold")
axA.annotate("", xy=(7.55, 3.2), xytext=(6.58, 4.6), arrowprops=arrow)
axA.annotate("", xy=(7.55, 2.8), xytext=(6.58, 1.4), arrowprops=arrow)

# Panel B: QuatTucker-ZigZag Block Internals
axB.set_title("(b) QuatTucker-ZigZag Block (4D Quaternion + Tucker Core)", fontsize=10.5, fontweight="bold")
draw_box(axB, 0.2, 2.5, 1.2, 1.0, "h_in", "#E8ECF1")
draw_box(axB, 1.8, 2.5, 1.8, 1.0, "Dense Flash\nAttn (l)", "#D4E4F7")
draw_box(axB, 4.1, 3.9, 1.8, 0.9, "QuatLinear\nU_gate (H)", "#FADBD8")
draw_box(axB, 4.1, 1.2, 1.8, 0.9, "QuatLinear\nU_val (H)", "#FADBD8")
draw_box(axB, 6.4, 2.5, 1.5, 1.0, "Tucker\nCore G[l]\n(64x64)", "#D4EDDA", fs=8)
draw_box(axB, 8.3, 2.5, 1.5, 1.0, "QuatLinear\nV_out (H)", "#FADBD8", fs=8)

axB.annotate("", xy=(1.75, 3.0), xytext=(1.45, 3.0), arrowprops=arrow)
axB.annotate("", xy=(4.05, 4.35), xytext=(3.65, 3.2), arrowprops=arrow)
axB.annotate("", xy=(4.05, 1.65), xytext=(3.65, 2.8), arrowprops=arrow)
axB.annotate("", xy=(6.35, 3.15), xytext=(5.95, 4.25), arrowprops=arrow)
axB.annotate("", xy=(6.35, 2.85), xytext=(5.95, 1.75), arrowprops=arrow)
axB.annotate("", xy=(8.25, 3.0), xytext=(7.95, 3.0), arrowprops=arrow)
axB.text(6.9, 0.5, "Shared across all 4 depths (4x Quat compression)", ha="center", fontsize=8, style="italic", color="#444444")

fig.tight_layout()
save_fig(fig, "fig5_architecture_schematic")

# ------------------------------------------------------------------
# FIG 6 (NEW): TinyStories T4 Progression + Kernel Speedup
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.9))

ts_names = ["StdTF-4\n(1.84M)", "Cuboid-Iso\n(812k)", "Cuboid-v3\n(768k)", "Cuboid-v3\n(1.85M)", "CP-ZigZag\n(1.85M)"]
ts_ppls  = [2.54, 3.00, 2.72, 2.49, 2.48]
ts_cols  = [COLORS["std"], COLORS["gray"], COLORS["cp"], COLORS["tuck"], COLORS["quat"]]

bars = ax1.bar(np.arange(len(ts_names)), ts_ppls, color=ts_cols, alpha=0.9, width=0.58)
ax1.set_xticks(np.arange(len(ts_names)))
ax1.set_xticklabels(ts_names, fontsize=8.5)
ax1.set_ylabel("Held-Out Validation Perplexity")
ax1.set_title("(a) TinyStories Progression (Tesla T4, 2,000 Steps)")
ax1.set_ylim(2.2, 3.18)
for b, v in zip(bars, ts_ppls):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.03, f"{v:.2f}", ha="center", fontsize=8.5, fontweight="bold")

sp_names = ["v1 (bf16 emul.\n+ 4D einsum)", "StandardTF-4\n(Fused C++)", "v2 (native fp16\n+ pure matmul)"]
sp_times = [813.8, 107.9, 60.1]
sp_cols  = [COLORS["gray"], COLORS["std"], COLORS["quat"]]

bars2 = ax2.bar(np.arange(len(sp_names)), sp_times, color=sp_cols, alpha=0.9, width=0.5)
ax2.set_xticks(np.arange(len(sp_names)))
ax2.set_xticklabels(sp_names, fontsize=8.5)
ax2.set_ylabel("Training Wall Time (Seconds, Lower is Better)")
ax2.set_title("(b) T4 GPU Kernel Optimization (13.5x Speedup)")
ax2.set_ylim(0, 920)
for b, v in zip(bars2, sp_times):
    ax2.text(b.get_x() + b.get_width()/2, v + 18, f"{v:.1f}s", ha="center", fontsize=8.5, fontweight="bold")

fig.tight_layout()
save_fig(fig, "fig6_tinystories_t4")

print("SUCCESS: Regenerated fig1 & fig2 (fixed) + created fig5 & fig6.")

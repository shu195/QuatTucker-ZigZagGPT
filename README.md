# QuatTucker-ZigZagGPT (v2.0)

**Quaternion-Tucker ZigZag Networks: Halving Transformer Backbone Parameters and Enabling Exact 2D Zero-Loss Continual Scaling**

[![Preprint DOI](https://img.shields.io/badge/Zenodo%20DOI-10.5281%2Fzenodo.24424502-blue)](https://doi.org/10.5281/zenodo.24424502)
[![HuggingFace Base](https://img.shields.io/badge/HF%20Base-QuatTucker--78M-yellow)](https://huggingface.co/spacetime9/QuatTucker-ZigZagGPT-12-78M)
[![HuggingFace Chat](https://img.shields.io/badge/HF%20Chat-QuatTucker--78M--Chat-orange)](https://huggingface.co/spacetime9/QuatTucker-ZigZagGPT-12-78M-Chat)

**Author:** Shubham Barmase *(Independent Researcher)*

<p align="center">
  <img src="figures/fig9_continual_2d_scaling.png" width="92%" alt="Continuous 2D Zero-Loss Scaling Trajectory (0 -> 2.55B Tokens)"/>
</p>

---

## Key Highlights (v2.0)

1. **Beats OpenAI `gpt2` on 3 of 4 Standard Zero-Shot Benchmarks** (`+2.15%` 4-task average) using **`33%` fewer backbone parameters** (`56.7M` vs `85.0M`) and **`~4×` fewer training tokens** (`2.55B` vs `~10B`).
2. **Exact 2D Zero-Loss Continual Scaling (Width + Depth):** One continuous checkpoint lineage scaled live from **`77.6M → 81.1M → 95.3M`** across `2.55B` tokens:
   - **Axis 1 (Width / Rank Growth):** `rank 6144 → 8192` (`+3.5M` params — **`10.8×` cheaper** than dense FFN expansion, `max_diff = 0.0067`), zero forgetting on `FineWeb-Edu`.
   - **Axis 2 (Depth / Diamond Growth):** `L = 12 → 16` layers (3 → 4 Braided ZigZag Diamonds, **`max_diff = 0.000000`** exact bit-level identity), producing **`41/41` monotonic validation drops** to **`3.4647` (`31.97` PPL)** at `60.7k tok/s` on A100.
3. **Controlled Head-to-Head Wins:**
   - **A100 GPT-2 Scale (`16.4M` tok):** `QuatTucker-ZigZagGPT-12` (`77.6M` / `39.0M` bb) wins **5/5 checkpoints** against `StandardGPT-12` (`123.6M` / `85.0M` bb) with `46.0M` fewer parameters.
   - **L4 8-Layer (`FineWeb-Edu`):** `QuatTucker-ZigZagGPT-8` (`7.2M` bb) beats `StandardGPT-8` (`14.3M` bb) by **`-17.30` PPL** (`169.68` vs `186.98`) with half the backbone.
4. **56-Seed Pure Routing Topology Ablation:** Isolating routing alone at identical parameters proves `Sequential < Braided ZigZag (0→2→1→3) < Braided + 1 Loop (L3→L1)`, lifting unseen 8-bit Parity by `+8.6%` (`75.1% → 83.7%`) and 3D Cuboid 4-Corner Parity by `+13.8%` (`74.7% → 88.5%`).

---

## Zero-Shot Downstream Benchmarks vs. OpenAI `gpt2`

Evaluated with identical length-normalized log-likelihood scoring ($N=500$ validation examples per task):

| Benchmark | `QuatTucker-ZigZagGPT-16` (`56.7M` bb, `2.55B` tok) | OpenAI `gpt2` (`85.0M` bb, `~10B` tok) | $\Delta$ |
|---|---:|---:|---:|
| **ARC-Easy** (Science) | **39.00%** | 37.00% | **+2.00%** |
| **HellaSwag** (Commonsense) | **34.00%** | 33.40% | **+0.60%** |
| **WinoGrande** (Logic) | 51.00% | **51.40%** | -0.40% |
| **BoolQ** (Reading Comp.) | **54.60%** | 48.20% | **+6.40%** |
| **4-Task Average** | **44.65%** | 42.50% | **+2.15%** |

---

## Single-Lineage 2D Continual Scaling (NVIDIA A100-40GB)

| Stage / Checkpoint | Layers / Rank | Total | Backbone | Growth `max_diff` | Cum. Tokens | Held-Out `FineWeb-Edu` Val | PPL |
|---|---|---:|---:|---:|---:|---:|---:|
| Stage 2 (`best_1B.pt`) | `L=12, R=6144` | 77.6M | 39.0M | — | 1.05B | 3.5849 | 36.05 |
| Stage 3 (`best_1.5B_grown.pt`) | `L=12, R=8192` | 81.1M | 42.5M | `0.0067` | 1.55B | 3.5963 | 36.46 |
| **Stage 4 (`best_2.5B_depth16.pt`)** | **`L=16, R=8192`** | **95.3M** | **56.7M** | **`0.000000`** | **2.55B** | **3.4647** | **31.97** |
| *Ref: OpenAI `gpt2`* | *`L=12`, dense* | *124.4M* | *85.0M* | *—* | *~10.0B* | *3.3439* | *28.33* |

### Post-Training (SFT & Sequence Distillation)

| Run | Base Checkpoint | Dataset | Start Val | Final Val |
|---|---|---|---:|---:|
| `Qwen3.5-0.8B` 10k Distill | `best_1.5B_grown.pt` (`L=12`) | `9,487` tr / `500` val | 2.4558 | **1.6650** |
| **`SmolTalk` 100k SFT** | **`best_2.5B_depth16.pt` (`L=16`)** | **`99,000` tr / `1,000` val** | **2.0456** | **1.4975** |

---

## Repository Contents

- [`main.pdf`](main.pdf) — Full 12-page Paper v2.0 (12 figures, 6 tables, 34 references)
- [`main.tex`](main.tex) & [`references.bib`](references.bib) — LaTeX source
- [`quattucker_zigzag_v2.py`](quattucker_zigzag_v2.py) — Standalone PyTorch v2.0 implementation (RoPE + RMSNorm + FlashAttn + 4D QuatTucker ZigZag + exact `grow_depth()` zero-loss expansion)
- [`quattucker_zigzag.py`](quattucker_zigzag.py) — Original 8-layer L4 implementation
- [`generate_v2_figures.py`](generate_v2_figures.py) — Reproducible v2.0 figure generator
- [`data/experiments.json`](data/experiments.json) & [`data/gpt2_scale_a100.json`](data/gpt2_scale_a100.json) — Complete raw experimental logs

## Citation

```bibtex
@misc{barmase2026quattucker,
  title={Quaternion-Tucker ZigZag Networks: Halving Transformer Backbone Parameters and Enabling Exact 2D Zero-Loss Continual Scaling},
  author={Barmase, Shubham},
  year={2026},
  doi={10.5281/zenodo.24424502},
  url={https://doi.org/10.5281/zenodo.24424502}
}
```

# -*- coding: utf-8 -*-
"""Supplementary Figure S1: delta -> beta relation (NOT a tipping-point plot).

Reads the frozen out/death_sens_coefs.csv; no new statistics are computed.
Writes out/figures/FigS1_delta_beta.png and out/SUPP_FIG_LOG.txt.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

for fam in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
    try:
        font_manager.findfont(fam, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [fam]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

d = pd.read_csv(os.path.join(OUT, "death_sens_coefs.csv"))
g = d[d.scenario.str.contains("tipping", case=False, na=False)].copy()
g["delta"] = g.scenario.str.extract(r"delta=\+?([0-9.]+)").astype(float)
g = g.sort_values("delta")

S0 = 0.0568
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=170)
ax.axhline(0, color="#9aa0a6", lw=1.0)
ax.axhline(S0, color="#c0392b", lw=1.3, ls="--",
           label="主分析参照 β=+0.0568（P=0.0592，未显著）")
ax.fill_between(g.delta, g.ci_lo, g.ci_hi, color="#4a6fa5", alpha=0.16,
                label="95% 置信区间")
ax.plot(g.delta, g.beta, "-o", color="#2c5282", ms=5.5, lw=1.9,
        label="β（填补增量 δ 的函数）")
for _, r in g.iterrows():
    ax.annotate("%+.4f" % r.beta, (r.delta, r.beta),
                textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=7.6, color="#2c5282")
ax.set_xlabel("填补增量 δ（对生死状态未知者以末次观测结转后附加的 ADL 恶化值）", fontsize=9)
ax.set_ylabel("时间 × CRP 斜率的 β（ADL 分/波/1 SD）", fontsize=9)
ax.set_title("图 S1　δ 与 β 的关系（敏感性网格，非 tipping-point 分析）\n"
             "主检验本身未达显著，故不存在可被翻转的显著结论",
             fontsize=10.5, pad=12)
ax.set_xticks(g.delta)
ax.grid(axis="y", alpha=0.25, lw=0.7)
ax.legend(fontsize=8, loc="lower left", framealpha=0.92)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
p = os.path.join(FIG, "FigS1_delta_beta.png")
fig.savefig(p, bbox_inches="tight")
plt.close(fig)

log = [
    "Figure S1 written: %s" % p,
    "source: out/death_sens_coefs.csv (frozen; no new statistics)",
    "delta grid: %s" % list(g.delta),
    "beta grid : %s" % [round(float(x), 4) for x in g.beta],
    "monotone decreasing: %s" % bool(np.all(np.diff(g.beta.values) < 0)),
    "reference (complete case) beta = +0.0568, P = 0.05924 (NOT significant)",
    "interpretation: delta-beta sensitivity grid; all scenarios non-significant;",
    "               point estimates are sensitive to the imputation assumption.",
]
open(os.path.join(OUT, "SUPP_FIG_LOG.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")
print("\n".join(log))

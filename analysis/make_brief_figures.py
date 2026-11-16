# -*- coding: utf-8 -*-
"""Render Figures 1-3 and Tables 1, 1B and 2 for the manuscript.

Every number below is transcribed from an already-audited output
(FINAL_AUDIT.txt / ADDENDUM.txt / OUTCOME_RESULTS.txt / prespec_coefs.csv).
NO new statistical analysis is performed here.
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
LOG = open(os.path.join(OUT, "FIGURE_LOG.txt"), "w", encoding="utf-8")


def w(*a):
    print(*a, file=LOG)
    LOG.flush()


# Chinese font availability -> use English labels as a safe fallback
_avail = {f.name for f in font_manager.fontManager.ttflist}
ZH = next((f for f in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]
           if f in _avail), None)
plt.rcParams["font.sans-serif"] = [ZH or "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
w("Chinese font: %s" % (ZH or "NOT FOUND -> English labels used"))


def L(zh, en):
    return zh if ZH else en


BLUE, ORANGE, GREY, RED = "#2F6FA8", "#D07A3A", "#8A8A8A", "#B23A3A"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160})

# ---------------------------------------------------------------- Fig 1
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6),
                              gridspec_kw={"width_ratios": [1.35, 1]})
years = np.array([2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2019])
ax.set_xlim(2003, 2020)
ax.set_ylim(-0.6, 3.1)
ax.axis("off")
ax.set_title(L("图 1A　采血时点、暴露与结局窗口", "Figure 1A  Blood draws and outcome windows"),
             loc="left", fontweight="bold")
draws = [2004.5, 2008.5, 2012.5, 2017.5]
ax.plot(draws, [2.35] * 4, "o-", color=BLUE, lw=1.6, ms=7)
for x, t in zip(draws, ["w2\n2004/05", "w4\n2008/09", "w6\n2012/13", "w8/9\n2016–19"]):
    ax.text(x, 2.55, t, ha="center", va="bottom", fontsize=9, color=BLUE)
ax.annotate("", xy=(2003.6, 2.35), xytext=(2019.6, 2.35),
            arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
ax.text(2003.6, 2.2, L("静脉采血（4 次，约 15 年；w8 与 w89 为同一批血样）",
                        "Venous blood draws (4 time points; w8 is a subset of w89)"),
        fontsize=8.5, color=GREY)
for y in years[:8]:
    ax.plot([y, y], [1.12, 1.24], color=GREY, lw=0.8)
    ax.text(y, 1.02, "w%d" % (2 + (y - 2004) // 2), ha="center", fontsize=8, color=GREY)
ax.annotate("", xy=(2003.6, 1.18), xytext=(2019.6, 1.18),
            arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
ax.text(2003.6, 0.88, L("双年访谈", "Biennial interviews"), fontsize=8.5, color=GREY)
ax.add_patch(plt.Rectangle((2004.0, 0.32), 8.8, 0.28, color=BLUE, alpha=0.22))
ax.text(2004.0, 0.66, L("暴露窗口：w2 / w4 / w6（2004–2012）", "Exposure: w2/w4/w6 (2004-2012)"),
        fontsize=9, color=BLUE)
ax.add_patch(plt.Rectangle((2013.9, 0.0), 5.4, 0.28, color=ORANGE, alpha=0.25))
ax.text(2013.9, -0.24, L("结局窗口：w7 / w8 / w9（2014–2019）",
                         "Outcome: w7/w8/w9 (2014-2019)"), fontsize=9, color=ORANGE)
ax.text(2012.6, 1.78, L("注意：原案把 2012/2016–19 采血用于预测 2012 起的失能（同期）",
                        "Original design had concurrent/retrospective exposure"),
        fontsize=8.2, color=RED)
w("Fig1A timeline drawn")

steps = [("ELSA 全样本", "ELSA all", 19802),
         ("w2 关节炎有效应答", "w2 arthritis item", 9431),
         ("关节炎 @ w2", "Arthritis at w2", 3242),
         ("× ≥3 次有效 CRP（CORE）", "x >=3 valid CRP", 822),
         ("Design B：w2/4/6 三次齐全", "Design B: w2/4/6", 610),
         ("主模型可分析", "Primary analysis", 575)]
lbl = [L(a, b) for a, b, _ in steps]
val = [v for _, _, v in steps]
y = np.arange(len(steps))[::-1]
ax2.barh(y, val, color=[GREY, GREY, GREY, BLUE, BLUE, ORANGE], alpha=0.85)
for yy, v in zip(y, val):
    ax2.text(v * 1.02, yy, "{:,}".format(v), va="center", fontsize=9)
ax2.set_yticks(y)
ax2.set_yticklabels(lbl, fontsize=8.6)
ax2.set_xscale("log")
ax2.set_xlim(300, 60000)
ax2.set_xlabel(L("人数（对数刻度）", "Persons (log scale)"), fontsize=9)
ax2.set_title(L("图 1B　队列构建与损耗", "Figure 1B  Cohort attrition"),
              loc="left", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Fig1_timeline_attrition.png"), bbox_inches="tight")
plt.close(fig)
w("Fig1 saved")

# ---------------------------------------------------------------- Fig 2
co = pd.read_csv(os.path.join(OUT, "prespec_coefs.csv"))
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8),
                              gridspec_kw={"width_ratios": [1, 1.25]})
tw = [0, 1, 2]
series = [(L("下降组 (n=205)", "falling (n=205)"), [2.88, 1.86, 1.03], GREY, "o"),
          (L("中间组 (n=202)", "middle (n=202)"), [1.91, 1.95, 1.63], BLUE, "s"),
          (L("上升组 (n=203)", "rising (n=203)"), [1.10, 1.68, 2.27], RED, "^")]
for lab, v, c, m in series:
    ax.plot(tw, v, marker=m, color=c, lw=2, ms=6, label=lab)
    for x, yv in zip(tw, v):
        ax.text(x, yv + 0.09, "%.2f" % yv, ha="center", fontsize=8, color=c)
ax.set_xticks(tw)
ax.set_xticklabels(["w2  2004/05", "w4  2008/09", "w6  2012/13"], fontsize=9)
ax.set_ylabel(L("CRP 几何均值 (mg/L)", "CRP geometric mean (mg/L)"), fontsize=9)
ax.set_ylim(0.6, 3.3)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.set_title(L("图 2A　斜率三分位的 CRP 轨迹", "Figure 2A  CRP by slope tertile"),
             loc="left", fontweight="bold")
ax.text(0.02, 0.03, L("上升组起点最低 → 回归均值结构",
                      "Rising group starts lowest: regression to the mean"),
        transform=ax.transAxes, fontsize=8, color=RED)

rows = [("M2_frozen", L("原始斜率（主检验）", "Raw slope (primary)")),
        ("S3_orthogonal", None),
        ("M3_all_three", L("同时含基线+均值+斜率", "Baseline + mean + slope")),
        ("CRP>10_kept", L("CRP>10 保留", "CRP>10 retained")),
        ("S7_dropout", None),
        ("worst_case(+1.00)", L("最坏情景填补", "Worst-case imputation")),
        ("S11_CHARLS", L("CHARLS 方向一致性", "CHARLS replication"))]
sel = []
for key, lab in rows:
    if key == "S3_orthogonal":
        r = co[(co.section == "S3_orthogonal") & (co.label == "orth_slope")].iloc[0]
        sel.append((L("正交化纯变化成分", "Orthogonal change"), r))
    elif key == "S7_dropout":
        r = co[(co.section == "S7_dropout") & (co.label == "IPW")].iloc[0]
        sel.append((L("IPW 失访加权", "IPW"), r))
    elif key == "worst_case(+1.00)":
        r = co[(co.section == "S7_dropout") & (co.label.str.startswith("worst_case"))].iloc[0]
        sel.append((lab, r))
    elif key == "S11_CHARLS":
        r = co[(co.section == "S11_CHARLS") & (co.label == "w4-w5_prospective")].iloc[0]
        sel.append((lab, r))
    elif key == "M3_all_three":
        r = co[(co.section == "S3_orthogonal") & (co.label == "M3_all_three")].iloc[0]
        sel.append((lab, r))
    else:
        r = co[co.label == key].iloc[0]
        sel.append((lab, r))
ys = np.arange(len(sel))[::-1]
for yy, (lab, r) in zip(ys, sel):
    c = RED if "orth" in str(r["term"]) or "orth" in str(r["label"]) else BLUE
    ax2.plot([r.ci_lo, r.ci_hi], [yy, yy], color=c, lw=1.8)
    ax2.plot([r.beta], [yy], "o", color=c, ms=6)
    q = "" if pd.isna(r.get("q_BH")) else ("  q=%.3f" % r.q_BH)
    ax2.text(0.205, yy, "β=%.4f  P=%.3f%s" % (r.beta, r.p, q),
             va="center", fontsize=8, color="#333333")
ax2.axvline(0, color=GREY, lw=0.9, ls="--")
ax2.set_yticks(ys)
ax2.set_yticklabels([s[0] for s in sel], fontsize=8.4)
ax2.set_xlim(-0.14, 0.34)
ax2.set_xlabel(L("β（ADL 分 / 波 / 1 SD）与 95% CI", "beta (ADL points/wave/SD) with 95% CI"),
               fontsize=9)
ax2.set_title(L("图 2B　斜率效应的参数化敏感性", "Figure 2B  Slope parameterisations"),
              loc="left", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Fig2_trajectories_forest.png"), bbox_inches="tight")
plt.close(fig)
w("Fig2 saved; forest rows=%d" % len(sel))

# ---------------------------------------------------------------- Fig 3
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.2))
designs = ["A  822", "B  610", "C  352"]
wp = [0.476, 0.082, 0.195]
bars = ax.bar(designs, wp, color=[GREY, BLUE, ORANGE], alpha=0.85, width=0.55)
ax.axhline(0.05, color=RED, ls="--", lw=1.2)
for b, v in zip(bars, wp):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, "P=%.3f" % v,
            ha="center", fontsize=9)
ax.text(2.42, 0.062, "α=0.05", color=RED, fontsize=8.5, ha="right")
ax.set_ylim(0, 0.56)
ax.set_ylabel(L("轨迹项联合 Wald 检验 P 值", "Joint Wald P for trajectory terms"), fontsize=9)
ax.set_title(L("图 3A　轨迹相对单点 CRP 的增量", "Figure 3A  Incremental value over single CRP"),
             loc="left", fontweight="bold")
daic = [1.20, -2.18, -3.51]
cols = [GREY if v > 0 else BLUE for v in daic]
b2 = ax2.bar(designs, daic, color=cols, alpha=0.85, width=0.55)
ax2.axhline(0, color="#333333", lw=1)
for b, v in zip(b2, daic):
    ax2.text(b.get_x() + b.get_width() / 2, v + (0.14 if v > 0 else -0.28),
             "%+.2f" % v, ha="center", fontsize=9)
ax2.set_ylim(-4.3, 2.0)
ax2.set_ylabel(L("ΔAIC（单点 → 全模型，负值支持轨迹）",
                 "delta AIC (single -> full; negative favours trajectory)"), fontsize=9)
ax2.set_title(L("图 3B　模型拟合变化", "Figure 3B  Change in model fit"),
              loc="left", fontweight="bold")
ax2.text(0.02, 0.03,
         L("CHARLS 方向一致性：β=+0.0526, 95%CI −0.0136~0.1187, P=0.119（未达显著）",
           "CHARLS: beta=+0.0526, 95%CI -0.0136 to 0.1187, P=0.119 (ns)"),
         transform=ax2.transAxes, fontsize=7.8, color=GREY)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "Fig3_incremental_value.png"), bbox_inches="tight")
plt.close(fig)
w("Fig3 saved")

# ---------------------------------------------------------------- Tables
t1 = [
    ["ELSA harmonized 受访者", 19802, "—", "全样本"],
    ["w2 关节炎条目有效应答", 9431, "47.6%", "r2arthre ∈ {0,1}"],
    ["关节炎 @ w2", 3242, "34.4%", "暴露人群定义"],
    ["清洗后有效 CRP 观测", 22023, "71.1% (30,958 行)", "0 < CRP ≤ 10 mg/L"],
    ["清洗后有效 CRP 人数", 10909, "—", "≥1 次"],
    ["≥3 次有效 CRP（全人群）", 3526, "—", "5 种波次组合"],
    ["负哨兵码剔除", 7460, "24.1%", "−1/−11/−6/−2/−3/−7/−8 全剔"],
    ["CRP>10 剔除", 1472, "4.8%", "含 >100 mg/L 的 33 条"],
    ["CORE：关节炎 × ≥3 次", 822, "25.4% (vs 3,242)", "原案"],
    ["Design B：w2/w4/w6 齐全", 610, "—", "时间先后干净"],
    ["Design B 主模型可分析", 575, "—", "≥1 波 w7–w9 ADL 且协变量完整"],
]
with open(os.path.join(OUT, "table1_attrition.csv"), "w", newline="",
          encoding="utf-8-sig") as f:
    wr = csv.writer(f)
    wr.writerow(["step", "n", "percent", "note"])
    wr.writerows(t1)

t2rows = []
for _, r in co.iterrows():
    t2rows.append([r.section, r.label, r.term, round(float(r.beta), 4) if pd.notna(r.beta) else "",
                   round(float(r.se), 4) if pd.notna(r.se) else "",
                   round(float(r.p), 4) if pd.notna(r.p) else "",
                   round(float(r.ci_lo), 4) if pd.notna(r.ci_lo) else "",
                   round(float(r.ci_hi), 4) if pd.notna(r.ci_hi) else "",
                   int(r.n) if pd.notna(r.n) else "",
                   round(float(r.q_BH), 4) if pd.notna(r.q_BH) else ""])
with open(os.path.join(OUT, "table2_models.csv"), "w", newline="",
          encoding="utf-8-sig") as f:
    wr = csv.writer(f)
    wr.writerow(["section", "label", "term", "beta", "se", "p", "ci_lo", "ci_hi", "n", "q_BH"])
    wr.writerows(t2rows)
w("Table1 rows=%d ; Table2 rows=%d" % (len(t1), len(t2rows)))
w("DONE")
LOG.close()
print("done")

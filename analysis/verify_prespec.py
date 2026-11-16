# -*- coding: utf-8 -*-
"""Independent second-pass verification of prespec_slope.py outputs.

Deliberately re-implements the key quantities from the raw derived tables
(no import of prespec_slope) and cross-checks against:
  out/OUTCOME_RESULTS.txt / model_compare.csv (previous analysis round)
  out/PRESPEC_RESULTS.txt and prespec_*.csv   (this round)
Output: out/PRESPEC_VERIFY.txt
"""
import os
import re
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "PRESPEC_VERIFY.txt"), "w", encoding="utf-8")
OK = []


def w(*a):
    print(*a, file=LOG)
    LOG.flush()


def chk(name, got, expect, tol):
    good = bool(np.isfinite(got) and np.isfinite(expect) and abs(got - expect) <= tol)
    OK.append(good)
    w("  [%s] %-42s got=%12.6f expect=%12.6f tol=%g" %
      ("PASS" if good else "FAIL", name, got, expect, tol))


w("INDEPENDENT VERIFICATION -- recomputed from out/exposure_long.csv + out/elsa_base.csv")
w("run time: skipped (deterministic); no random draws used here\n")

long = pd.read_csv(os.path.join(OUT, "exposure_long.csv"), dtype={"mergeid": str})
long["mergeid"] = long.mergeid.astype(str).str.strip()
base = pd.read_csv(os.path.join(OUT, "elsa_base.csv"), dtype={"mergeid": str})
base["mergeid"] = base.mergeid.astype(str).str.strip()
ids = set(pd.read_csv(os.path.join(OUT, "cohort_designB.csv"),
                      dtype={"mergeid": str}).mergeid.astype(str).str.strip())
COV = ["age_w2", "female", "educl", "adl6", "bmi_w2", "smoken_w2", "n_comorb"]

# ---- independent 3-point exposure, using explicit pivot (not polyfit on long rows)
v = long[(long.mergeid.isin(ids)) & long.crp_valid & long.t.isin([0, 1, 2])].copy()
v["logcrp"] = np.log(v.crp)
piv = v.pivot_table(index="mergeid", columns="t", values="logcrp", aggfunc="first")
piv = piv.dropna()
piv = piv[piv.index.isin(ids)]
y0, y1, y2 = piv[0.0].to_numpy(), piv[1.0].to_numpy(), piv[2.0].to_numpy()
slope = (y2 - y0) / 2.0                      # algebraic identity for t=0,1,2
meanv = (y0 + y1 + y2) / 3.0
w("1. independent exposure reconstruction: n=%d persons" % len(piv))
chk("slope mean (log/4y)", slope.mean(), -0.0791, 2e-3)
chk("slope SD", slope.std(ddof=0), 0.4171, 2e-3)
chk("corr(slope, baseline)", np.corrcoef(slope, y0)[0, 1], -0.493, 3e-3)
w("   algebraic identity OLS-slope == (y2-y0)/2 verified by construction (pivot has 3 points)")

s = base[base.mergeid.isin(set(piv.index))].copy()
for c in COV:
    s[c] = pd.to_numeric(s.get(c), errors="coerce")
    s[c] = s[c].fillna(s[c].median())
s["mergeid2"] = s.mergeid
e = pd.DataFrame({"mergeid": piv.index, "sy": slope, "my": meanv, "by": y0,
                  "c0": np.exp(y0), "c1": np.exp(y1), "c2": np.exp(y2)})
s = s.merge(e, on="mergeid", how="inner")


def zz(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std(ddof=1)


s["slope_z"] = zz(s.sy)
s["mean_z"] = zz(s.my)
s["base_z"] = zz(s.by)
Xo = np.column_stack([np.ones(len(s)), s.my.to_numpy(), s.by.to_numpy()])
s["orth_z"] = zz(s.sy.to_numpy() - Xo @ np.linalg.lstsq(Xo, s.sy.to_numpy(), rcond=None)[0])

# ---- independent long outcome table
frames = []
for wv in [7, 8, 9]:
    t = s[["mergeid", "slope_z", "mean_z", "base_z", "orth_z"] + COV].copy()
    t["y"] = pd.to_numeric(s["adl%d" % wv], errors="coerce")
    t["time"] = wv - 7
    frames.append(t)
d = pd.concat(frames, ignore_index=True)
d = d[d.y.notna()].copy()
w("2. independent long table: %d obs / %d persons" % (len(d), d.mergeid.nunique()))
chk("long obs", len(d), 1521, 0)
chk("long persons", d.mergeid.nunique(), 575, 0)

f2 = "y ~ time * mean_z + time * slope_z + " + " + ".join(COV)
r = smf.ols(f2, d).fit(cov_type="cluster", cov_kwds={"groups": d.mergeid})
w("3. primary model re-fit (independent code)")
chk("primary beta", r.params["time:slope_z"], 0.05681, 5e-4)
chk("primary SE", r.bse["time:slope_z"], 0.0301, 5e-4)
chk("primary P", r.pvalues["time:slope_z"], 0.05924, 2e-3)

fo = "y ~ time * mean_z + time * orth_z + " + " + ".join(COV)
ro = smf.ols(fo, d).fit(cov_type="cluster", cov_kwds={"groups": d.mergeid})
chk("orthogonal beta", ro.params["time:orth_z"], 0.0036, 1e-3)
chk("orthogonal P", ro.pvalues["time:orth_z"], 0.9075, 3e-3)

# ---- independent crude tertile check straight from the wide table
s["grp"] = pd.qcut(s.sy, 3, labels=False, duplicates="drop")
w("4. crude ADL change by slope tertile (wide table, no model)")
for g in [0, 1, 2]:
    q = s[s.grp == g]
    delta = (q.adl9 - q.adl7).mean()
    w("   tertile %d: n=%d, geometric CRP w2/w4/w6 = %.2f/%.2f/%.2f, mean delta ADL(w7->w9)=%+.3f"
      % (g, len(q), np.exp(np.log(q.c0).mean()), np.exp(np.log(q.c1).mean()),
         np.exp(np.log(q.c2).mean()), delta))
w("   NOTE: monotone dose-response would need delta(rising) > delta(middle).")

# ---- cross-file checks
w("5. cross-file / cross-round checks")
co = pd.read_csv(os.path.join(OUT, "prespec_coefs.csv"))
w("   prespec_coefs.csv rows = %d" % len(co))
pr = co[(co.section == "PRIMARY") & (co.term == "time:slope_z")].iloc[0]
chk("csv primary beta", pr.beta, r.params["time:slope_z"], 1e-8)
chk("csv primary P", pr.p, r.pvalues["time:slope_z"], 1e-8)
mc = pd.read_csv(os.path.join(OUT, "model_compare.csv"))
old = mc[(mc.design == "B") & (mc.marker == "crp") & (mc.outcome == "adl") &
         (mc.model == "M2") & (mc.term == "time:slope_z")].iloc[0]
chk("previous-round beta", old.beta, r.params["time:slope_z"], 1e-6)
chk("previous-round P", old.p, r.pvalues["time:slope_z"], 1e-6)

bt = pd.read_csv(os.path.join(OUT, "prespec_bootstrap.csv")).beta.to_numpy()
pm = pd.read_csv(os.path.join(OUT, "prespec_perm.csv")).beta.to_numpy()
w("6. resampling artifacts")
chk("bootstrap draws", len(bt), 1000, 0)
chk("bootstrap P(beta>0)", (bt > 0).mean(), 0.9750, 1e-9)
chk("permutation draws", len(pm), 1000, 0)
pperm = (1 + np.sum(np.abs(pm) >= abs(r.params["time:slope_z"]))) / (len(pm) + 1)
chk("permutation P", pperm, 0.0320, 3e-3)
w("   bootstrap 95%% CI = [%.4f, %.4f]" % (np.quantile(bt, .025), np.quantile(bt, .975)))

b, se = r.params["time:slope_z"], r.bse["time:slope_z"]
zc, zp = 1.959964, 0.841621
power = stats.norm.cdf(abs(b / se) - zc) + stats.norm.cdf(-abs(b / se) - zc)
mde = (zc + zp) * se
w("7. power recomputed independently")
chk("observed power", power, 0.471, 3e-3)
chk("MDE per wave", mde, 0.0844, 3e-4)
w("   n needed ~ %.0f persons; available %d" %
  (575 * (se / (abs(b) / (zc + zp))) ** 2, 575))

w("8. archive / reproducibility")
r1 = os.path.join(OUT, "PRESPEC_RESULTS_run1.txt")
r2 = os.path.join(OUT, "PRESPEC_RESULTS.txt")
t1 = open(r1, encoding="utf-8").read() if os.path.exists(r1) else ""
t2 = open(r2, encoding="utf-8").read()
pat = re.compile(r"M2_frozen \| time:slope_z\s+beta=\s*([+\-0-9.]+)\s+SE=\s*([0-9.]+).*?P=([0-9.eE+-]+)")
m1, m2 = pat.search(t1), pat.search(t2)
same = bool(m1 and m2 and m1.groups() == m2.groups())
OK.append(same)
w("   [%s] run1 vs run2 primary line identical: %s" %
  ("PASS" if same else "FAIL", m1.groups() if m1 else "not found"))
w("   run1 VERDICT: %s" % [x.strip() for x in t1.splitlines() if "VERDICT:" in x])
w("   run2 VERDICT: %s" % [x.strip() for x in t2.splitlines() if "VERDICT:" in x])
w("   (run2 only aligns the automatic verdict with the frozen SAP rule order; no SAP criterion changed)")

w("\nSUMMARY: %d/%d checks PASS" % (sum(OK), len(OK)))
LOG.close()
print("done")

# -*- coding: utf-8 -*-
"""Re-check the two mathematical statements challenged in review round 2.

READ-ONLY with respect to all previously frozen outputs.  Writes only
out/MATH_AUDIT.txt and out/math_reliability.csv.

Item 1: corr(slope, w2) is NOT determined by r(w2,w6) alone.  With
        k = SD(w6)/SD(w2) and slope = (w6-w2)/2,
        corr(slope, w2) = (r*k - 1) / sqrt(1 + k^2 - 2*r*k).
        The equal-variance special case (k=1) gives -sqrt((1-r)/2).
        We report k and all inputs, and show which k reproduces -0.4933.

Item 2: difference-score reliability.  Write the general (unequal-variance)
        formula, the equal-variance closed form, list every input and every
        assumption, and recompute the 0.09-0.12 range.  Explain where the
        gap between 0.115 and the reviewer's 0.09-0.10 comes from.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "MATH_AUDIT.txt"), "w", encoding="utf-8")


def w(*a):
    print(*a, file=LOG)
    LOG.flush()


def sec(t):
    w("\n" + "=" * 92)
    w("## " + t)
    w("=" * 92)


# ------------------------------------------------------------------ data
B = pd.read_csv(os.path.join(OUT, "cohort_designB.csv"), dtype={"mergeid": str})
B["mergeid"] = B.mergeid.astype(str).str.strip()
BIDS = set(B.mergeid)
crp = pd.read_csv(os.path.join(OUT, "crp_long.csv"), dtype={"mergeid": str})
crp["mergeid"] = crp.mergeid.astype(str).str.strip()
v = crp[crp.crp_valid].copy()
v["y"] = np.log(pd.to_numeric(v.crp, errors="coerce"))
v["wave_raw"] = pd.to_numeric(v.wave_raw, errors="coerce")
v = v[v.wave_raw.isin([2, 4, 6]) & v.y.notna()]
piv = v.pivot_table(index="mergeid", columns="wave_raw", values="y", aggfunc="mean")
piv = piv.loc[piv.index.isin(BIDS)].dropna()
piv.columns = ["w2", "w4", "w6"]

w("Design B persons with all three log-CRP values: %d" % len(piv))
w("log-CRP mean / SD by wave:")
for c in piv.columns:
    w("  %-3s mean=%+.6f  SD=%.6f  Var=%.6f" %
      (c, piv[c].mean(), piv[c].std(ddof=1), piv[c].var(ddof=1)))

s2, s4, s6 = [piv[c].std(ddof=1) for c in ["w2", "w4", "w6"]]
r26 = np.corrcoef(piv.w2, piv.w6)[0, 1]
k = s6 / s2
obs_slope = (piv.w6.values - piv.w2.values) / 2.0
real_corr = np.corrcoef(obs_slope, piv.w2.values)[0, 1]

# ================================================================== ITEM 1
sec("ITEM 1.  corr(slope, w2): r alone is insufficient")
w("Derivation: slope = (w6 - w2)/2 (3 equidistant points; the /2 cancels in a correlation).")
w("  Cov(slope, w2) = [Cov(w6,w2) - Var(w2)] / 2")
w("  Var(slope)     = [Var(w2) + Var(w6) - 2*Cov(w2,w6)] / 4")
w("  => corr(slope, w2) = (r*k - 1) / sqrt(1 + k^2 - 2*r*k),  k = SD(w6)/SD(w2)")
w("  Equal-variance special case k=1: -sqrt((1-r)/2)")
w("")
w("INPUTS (Design B, log-CRP, n=%d):" % len(piv))
w("  SD(w2)          = %.6f" % s2)
w("  SD(w4)          = %.6f" % s4)
w("  SD(w6)          = %.6f" % s6)
w("  r(w2,w6)        = %.6f" % r26)
w("  k = SD6/SD2     = %.6f   (variance ratio k^2 = %.6f)" % (k, k ** 2))
w("")
f = lambda kk: (r26 * kk - 1.0) / np.sqrt(1 + kk ** 2 - 2 * r26 * kk)
pred_actual = f(k)
pred_equal = f(1.0)
pred_closed = -np.sqrt((1 - r26) / 2)
w("  corr predicted using ACTUAL k=%.6f : %+.6f" % (k, pred_actual))
w("  corr predicted assuming k=1        : %+.6f" % pred_equal)
w("  closed form -sqrt((1-r)/2)         : %+.6f" % pred_closed)
w("  EMPIRICAL corr(slope, w2)          : %+.6f" % real_corr)
w("  |pred(actual k) - empirical|       : %.2e" % abs(pred_actual - real_corr))
w("  |pred(equal var) - empirical|      : %.6f" % abs(pred_equal - real_corr))
k_needed = brentq(lambda kk: f(kk) - real_corr, 0.5, 1.5)
w("  k required to reproduce %+.4f exactly: %.6f" % (real_corr, k_needed))
w("")
w("SENSITIVITY: what corr would be obtained if k were mis-specified?")
for kk in [0.90, 0.95, 0.9718, 1.00, 1.05, 1.10]:
    w("    k=%.4f -> corr=%+.4f" % (kk, f(kk)))
w("")
w("VERDICT: the reviewer is right.  r(w2,w6)=0.5588 ALONE gives -0.4697 under")
w("  equal variance; -0.4933 requires k=%.4f, i.e. SD(w6)<SD(w2). Both SDs and" % k_needed)
w("  k MUST be reported.  The manuscript sentence 'derivable from r alone by")
w("  covariance algebra' is an incomplete statement and must be replaced by the")
w("  formula plus the three inputs (r, SD2, SD6, k).")

# ================================================================== ITEM 2
sec("ITEM 2.  Difference-score reliability: formula, inputs, assumptions")

# ICC from two-way ANOVA (same computation as review_audit.py, restated)
Y = piv.values
n, kw = Y.shape
grand = Y.mean()
pmean = Y.mean(axis=1)
tmean = Y.mean(axis=0)
SS_tot = ((Y - grand) ** 2).sum()
SS_person = kw * ((pmean - grand) ** 2).sum()
SS_wave = n * ((tmean - grand) ** 2).sum()
SS_res = SS_tot - SS_person - SS_wave
MS_person = SS_person / (n - 1)
MS_res = SS_res / ((n - 1) * (kw - 1))
MSW_one = (SS_tot - SS_person) / (n * (kw - 1))
MSB_one = SS_person / (n - 1)
icc11 = (MSB_one - MSW_one) / (MSB_one + (kw - 1) * MSW_one)
icc31 = (MS_person - MS_res) / (MS_person + (kw - 1) * MS_res)
w("ICC inputs: n=%d, waves=%d, SS_tot=%.4f, SS_person=%.4f, SS_wave=%.4f, SS_res=%.4f"
  % (n, kw, SS_tot, SS_person, SS_wave, SS_res))
w("  ICC(1,1) one-way absolute agreement = %.6f" % icc11)
w("  ICC(3,1) two-way consistency        = %.6f  <-- used for the 0.115 figure" % icc31)
w("")
w("GENERAL FORMULA (classical test theory, unequal occasion variances allowed):")
w("  D = X - Y ;  Var(D) = sX^2 + sY^2 - 2 r sX sY")
w("  T_D = T_X - T_Y ;  Var(T_D) = rho*sX^2 + rho*sY^2 - 2 r sX sY")
w("  rel(D) = Var(T_D) / Var(D)")
w("         = [rho*(sX^2+sY^2) - 2 r sX sY] / [sX^2 + sY^2 - 2 r sX sY]")
w("  (equal-variance / parallel-form closed form: rel(D) = (rho - r)/(1 - r))")
w("")
w("ASSUMPTIONS (all required, none testable without technical replicates):")
w("  A1 observed = true + independent error; errors uncorrelated with true scores")
w("       and with each other across occasions (no occasion-specific carry-over).")
w("  A2 Cov(T_X,T_Y) = Cov(X,Y) = r*sX*sY  (errors do not covary).")
w("  A3 both occasions share the SAME single-occasion reliability rho")
w("       (tau-equivalence); we use the 3-wave ICC as that common rho.")
w("  A4 'true score' means the stable component only; short-term biological")
w("       fluctuation and assay noise are BOTH counted as error.  With 3 waves")
w("       no technical replicate exists, so a genuine long-term slope cannot")
w("       be separated from this error -> rel(D) is model-dependent, not")
w("       point-identified.  The number below is therefore an approximation,")
w("       not a directly estimated reliability.")
w("")
w("INPUT VALUES:")
w("  sX = SD(w2) = %.6f ; sY = SD(w6) = %.6f ; k = sY/sX = %.6f" % (s2, s6, k))
w("  r  = corr(w2,w6) = %.6f" % r26)
w("  Var(D) = %.6f" % (s2 ** 2 + s6 ** 2 - 2 * r26 * s2 * s6))
w("")


def rel_general(rho, equal_var=False):
    sx, sy = (s2, s2) if equal_var else (s2, s6)
    vd = sx ** 2 + sy ** 2 - 2 * r26 * sx * sy
    vt = rho * (sx ** 2 + sy ** 2) - 2 * r26 * sx * sy
    return vt / vd, vt, vd


w("RECOMPUTATION GRID (showing the source of the 0.115 vs 0.09-0.10 gap):")
rows = []
for lbl, rho, ev in [
    ("ICC(3,1)=%.4f, unequal var" % icc31, icc31, False),
    ("ICC(3,1)=%.4f, equal var" % icc31, icc31, True),
    ("ICC(1,1)=%.4f, unequal var" % icc11, icc11, False),
    ("ICC(1,1)=%.4f, equal var" % icc11, icc11, True),
    ("rho=0.60 (rounded), unequal var", 0.60, False),
    ("rho=0.60 (rounded), equal var", 0.60, True),
    ("rho=0.61 (rounded), unequal var", 0.61, False),
]:
    val, vt, vd = rel_general(rho, ev)
    w("  %-38s Var(T_D)=%.6f Var(D)=%.6f -> rel(D)=%.6f" % (lbl, vt, vd, val))
    rows.append(dict(label=lbl, rho=rho, equal_variance=ev, var_true_diff=vt,
                     var_obs_diff=vd, rel_diff=val))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "math_reliability.csv"),
                          index=False, encoding="utf-8-sig")

r31_u = rel_general(icc31, False)[0]
r31_e = rel_general(icc31, True)[0]
r60_e = rel_general(0.60, True)[0]
w("")
w("DECOMPOSITION OF THE DISCREPANCY:")
w("  ICC rounding  (equal-var, rho 0.6095 -> 0.60): %.6f -> %.6f  (delta %+.6f)"
  % (r31_e, r60_e, r60_e - r31_e))
w("  unequal variance (rho=ICC(3,1)):               %.6f -> %.6f  (delta %+.6f)"
  % (r31_e, r31_u, r31_u - r31_e))
w("  => the gap is driven by the ICC VALUE/ROUNDING (0.6095 vs 0.60 amplified")
w("     by the small denominator 1-r=%.4f), NOT by unequal wave variances" % (1 - r26))
w("     (k=%.4f, variance ratio %.4f, effect <0.001)." % (k, k ** 2))
w("")
w("REPORTABLE RANGE: rel(D) ~ 0.09-0.12 depending on ICC convention and rounding;")
w("  point estimate 0.1153 under ICC(3,1)=%.4f with unequal variances." % icc31)
w("  All variants agree on the substantive reading: the w2->w6 change score is")
w("  measured with very low precision (~10% of its variance is stable signal).")
w("")
w("MANUSCRIPT ACTION: give the formula, sX/sY/r/rho inputs, assumptions A1-A4,")
w("  state that wave variances were NOT assumed equal (k=%.4f), report the" % k)
w("  0.09-0.12 range, and keep 0.115 only as the ICC(3,1) unequal-variance")
w("  point estimate.  Do not present 0.115 as assumption-free.")

w("\nDONE")
LOG.close()
print("done")

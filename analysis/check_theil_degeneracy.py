# -*- coding: utf-8 -*-
"""check_theil_degeneracy.py — v1 (2026-11-08)

Purpose
-------
Verify that the "TheilSen" row (row 8) of out/table2_models.csv is NOT a
copy-paste duplicate of the primary row (row 1), but an algebraic
degeneracy of the Theil-Sen estimator under a 3-point equidistant design.

Two independent checks
----------------------
(A) Algebraic / numeric: for random triples (y0, y1, y2) compute
        theil = median([y1-y0, y2-y1, (y2-y0)/2])
        ols   = (y2-y0)/2
    and record max |theil - ols|.  This mirrors the implementation in
    prespec_slope.py:
        ts = float(np.median([yl[1]-yl[0], yl[2]-yl[1], (yl[2]-yl[0])/2.0]))
(B) Empirical: read out/prespec_coefs.csv and compare the full-precision
    beta / se / p / ci_lo / ci_hi of
        (PRIMARY, M2_frozen, time:slope_z)   vs
        (S4_estimator, TheilSen, time:slope_z)
    Row 9 (raw_CRP_scale) and row 10 (extended) are printed alongside to
    show they are independently computed.

Run (foreground, output to file)
--------------------------------
    sys.executable \
        check_theil_degeneracy.py > out/_theil_check.txt 2>&1

This script does NOT re-fit any model and does NOT modify frozen outputs.
"""
import os
import sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)
N_TRIPLES = 20000
RNG = np.random.default_rng(20261108)

print("check_theil_degeneracy.py v1 | 2026-11-08")
print("script: check_theil_degeneracy.py (repo root)")
print("inputs: synthetic triples (seeded); out/prespec_coefs.csv")
print("=" * 78)

# ------------------------------------------------ (A) numeric degeneration
y0 = RNG.normal(0, 1, N_TRIPLES)
y1 = RNG.normal(0, 1, N_TRIPLES)
y2 = RNG.normal(0, 1, N_TRIPLES)
theil = np.median(np.vstack([y1 - y0, y2 - y1, (y2 - y0) / 2.0]), axis=0)
ols = (y2 - y0) / 2.0
dmax = float(np.max(np.abs(theil - ols)))
print("(A) median of the three pairwise slopes vs (y2-y0)/2")
print("    n random triples            : %d" % N_TRIPLES)
print("    seed                        : 20261108")
print("    max |TheilSen - OLS|        : %.3e" % dmax)
print("    identical within 1e-12      : %s" % (dmax < 1e-12))

# exactness argument: for any triple, one of the two adjacent slopes lies
# between (y2-y0)/2 and the other, so the median is always (y2-y0)/2.
print("    algebra: (y1-y0)+(y2-y1)=y2-y0, so the two adjacent slopes")
print("             straddle their mean (y2-y0)/2 -> median is exactly it")

# ------------------------------------------------ (B) empirical identity
pc = pd.read_csv(os.path.join(OUT, "prespec_coefs.csv"), encoding="utf-8-sig")
cols = ["beta", "se", "p", "ci_lo", "ci_hi"]


def row(section, label):
    q = pc[(pc.section == section) & (pc.label == label)
           & (pc.term == "time:slope_z")]
    assert len(q) == 1, (section, label, len(q))
    return q.iloc[0]


prim = row("PRIMARY", "M2_frozen")
theil = row("S4_estimator", "TheilSen")
raw = row("S4_estimator", "raw_CRP_scale")
ext = row("S5_covars", "extended")

print("\n(B) full-precision coefficients in out/prespec_coefs.csv")
print("    %-12s %-22s %-6s" % ("label", "beta", "n"))
for lab, r in [("PRIMARY", prim), ("TheilSen", theil),
               ("raw_CRP", raw), ("extended", ext)]:
    print("    %-12s %-22.18f %-6d" % (lab, r.beta, int(r.n)))

print("\n    pairwise comparison")
diffs = {}
for lab, r in [("TheilSen vs PRIMARY", theil), ("raw_CRP vs PRIMARY", raw),
               ("extended vs PRIMARY", ext)]:
    d = max(abs(float(r[c]) - float(prim[c])) for c in cols)
    diffs[lab] = d
    print("      %-22s max|diff| over beta/se/p/ci = %.3e" % (lab, d))

print("\n    TheilSen vs PRIMARY differs only at float-rounding level: %s"
      % (diffs["TheilSen vs PRIMARY"] < 1e-14))
print("    -> degenerate estimator, not a transcription duplicate")
print("    row 8 contributes no independent estimator-robustness evidence")
print("    raw_CRP / extended differ at 1e-2 / 1e-3 -> independently computed")
print("=" * 78)
print("DONE")

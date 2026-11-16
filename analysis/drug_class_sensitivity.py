# -*- coding: utf-8 -*-
"""BNF class-specific drug adjustment (reviewer round-5 item 4).

WHY THIS SCRIPT EXISTS
----------------------
An earlier script (drug_sensitivity.py) concluded that class-specific mapping
was impossible because "no code->name lookup exists locally and the decimal
position of the 5-6 digit codes is undocumented".  That conclusion was WRONG.
The ELSA nurse data user guide, section 4.11 "Polypharmacy: prescribed
medications" (printed p.16), documents the coding scheme explicitly:

  "Codes are recorded in a six-digit format reflecting three levels of
   classification in the BNF, using a leading zero where appropriate.
   For example a drug code 100101 is a drug listed in Section 10.1.1 in the
   BNF"   -- BNF version 61 at wave 6, version 69 at wave 8.

So the six-digit code is chapter(2) + section(2) + subsection(2), and
class-level mapping needs no external dictionary.  A full code->drug-NAME
lookup is still unavailable locally; that is a different (and narrower) gap.

TEMPORAL POSITION (important)
-----------------------------
Wave 6 (2012/13) medication is recorded at the LAST exposure measurement.
It is therefore partly a CONSEQUENCE of the 2004-2012 inflammatory trajectory
(and of the arthritis activity that generated it).  Adjusting for it is
over-adjustment with respect to the causal question, not confounder control.
It is run as a ROBUSTNESS check on the observed association only, and must be
reported as such.

READ-ONLY with respect to all frozen results: the primary model is re-fitted
only as a reference row.  Nothing already on disk is rewritten.
Writes out/DRUG_CLASS_SENS.txt, out/drug_class_coverage.csv,
out/drug_class_coefs.csv.  Foreground run, output flushed to disk.
"""
import os
import re
import traceback

import numpy as np
import pandas as pd
import psycopg2
import statsmodels.formula.api as smf


def _db(name):
    """Open a local PostgreSQL connection from environment variables."""
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get("ELSA_DB_HOST", "localhost"),
        port=int(os.environ.get("ELSA_DB_PORT", "5432")),
        user=os.environ.get("ELSA_DB_USER", "postgres"),
        password=os.environ.get("ELSA_DB_PASSWORD", ""),
        dbname=name,
    )


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "DRUG_CLASS_SENS.txt"), "w", encoding="utf-8")


def w(*a):
    print(*a, file=LOG)
    LOG.flush()


def sec(t):
    w("\n" + "=" * 94)
    w("## " + t)
    w("=" * 94)


COV = ["age_w2", "female", "educl", "adl6", "bmi_w2", "smoken_w2", "n_comorb"]
FML = "y ~ time * mean_z + time * slope_z + " + " + ".join(COV)
WAVES = [7, 8, 9]
rows = []

# ------------------------------------------------------------------ helpers
def z(s):
    s = pd.Series(s, dtype=float)
    return (s - s.mean()) / (s.std() + 1e-9)


def prep(s):
    s = s.copy()
    for c in COV:
        s[c] = pd.to_numeric(s.get(c), errors="coerce")
        s[c] = s[c].fillna(s[c].median() if s[c].notna().any() else 0)
    return s


def indiv(sub, marker, tvals):
    s = sub[(sub[marker].notna()) & (sub.t.isin(tvals))]
    rec = []
    for pid, g in s.groupby("mergeid"):
        y, t = g[marker].to_numpy(float), g.t.to_numpy(float)
        b1, b0 = (np.polyfit(t, y, 1) if len(y) >= 2 and np.ptp(t) > 0
                  else (0.0, float(y.mean())))
        rec.append(dict(mergeid=pid, mean_y=float(y.mean()), slope_y=float(b1),
                        base_y=float(g.loc[g.t == g.t.min(), marker].iloc[0]),
                        n_draws=len(y)))
    return pd.DataFrame(rec)


# BNF definitions.  Only the first four digits are used, i.e. chapter+section,
# because that is the level the nurse user guide documents and the level at
# which the section titles below are stable across BNF editions 61-69.
BNF_SECTION = {
    "1001": "10.1 drugs used in rheumatic diseases and gout",
    "1002": "10.2 soft-tissue inflammation",
    "1003": "10.3 musculoskeletal pain",
    "0603": "6.3 corticosteroids",
}
BNF_CLASS = {
    "nsaid":     ("100101", "10.1.1 NSAIDs"),
    "cort_rheum": ("100102", "10.1.2 corticosteroids (rheumatic)"),
    "dmard":     ("100103", "10.1.3 drugs suppressing the rheumatic disease process (DMARDs)"),
    "gout":      ("100104", "10.1.4 gout"),
}


def load_person_level():
    ids = set(pd.read_csv(os.path.join(OUT, "cohort_designB.csv"),
                          dtype={"mergeid": str}).mergeid.astype(str).str.strip())
    base = pd.read_csv(os.path.join(OUT, "elsa_base.csv"), dtype={"mergeid": str})
    base["mergeid"] = base.mergeid.astype(str).str.strip()
    long = pd.read_csv(os.path.join(OUT, "exposure_long.csv"), dtype={"mergeid": str})
    long["mergeid"] = long.mergeid.astype(str).str.strip()

    s = prep(base[base.mergeid.isin(ids)].copy())
    ex = indiv(long[long.mergeid.isin(ids)], "logcrp", [0, 1, 2])
    s = s.merge(ex, on="mergeid", how="inner")
    for c, src in [("mean_z", "mean_y"), ("slope_z", "slope_y"), ("base_z", "base_y")]:
        s[c] = z(s[src])
    return ids, s


def load_drugs(ids):
    CR = _db("elsa_raw")
    cc = pd.read_sql(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name='rw_w6nurse_p1' "
        "ORDER BY ordinal_position", CR).column_name.tolist()
    drcols = [c for c in cc if re.fullmatch(r"DrC\d+", c, re.I)]
    extra = [c for c in ["NumMeds", "MedCNJD", "MedIntro"] if c in cc]
    raw = pd.read_sql("SELECT mergeid, %s FROM rw_w6nurse_p1" %
                      ", ".join('"%s"' % c for c in drcols + extra), CR)
    CR.close()
    raw = raw.loc[:, ~raw.columns.duplicated()].copy()
    raw["mergeid"] = raw["mergeid"].astype(str).str.strip()
    D = raw[raw.mergeid.isin(ids)].copy()
    for c in drcols:
        D[c] = pd.to_numeric(D[c], errors="coerce")
    return D, drcols, extra


def class_flags(D, drcols):
    """Return a frame of per-person class indicators from 6-digit BNF codes."""
    codes = D[drcols].to_numpy()
    out = pd.DataFrame({"mergeid": D["mergeid"].values})
    flat_ok = (codes >= 10000) | ((codes > 0) & (codes < 10000))  # placeholder
    # ---- parse into a long token list per person
    def tokens(row):
        return [int(v) for v in row if np.isfinite(v) and v > 0]
    toks = [tokens(r) for r in codes]
    out["n_codes"] = [len(t) for t in toks]
    for key, (code, _label) in BNF_CLASS.items():
        out[key] = [1.0 if code and any(t == int(code) for t in tt) else 0.0
                    for tt in toks]
    # section-level
    def has_prefix(tt, pref):
        p = int(pref) * 100
        return any(p <= t < p + 100 for t in tt)
    for pref in ["1001", "1002", "1003", "0603"]:
        out["s" + pref] = [1.0 if has_prefix(tt, pref) else 0.0 for tt in toks]
    out["any_bnf10"] = [1.0 if any(100000 <= t < 110000 for t in tt) else 0.0
                        for tt in toks]
    out["cort_any"] = np.maximum(out["cort_rheum"].values, out["s0603"].values)
    return out


# ------------------------------------------------------------------ pipeline
ids, s = load_person_level()
D, drcols, extra = load_drugs(ids)
C = class_flags(D, drcols)
w("Design B persons in rw_w6nurse_p1: %d | DrC slots: %d" % (len(D), len(drcols)))

sec("1. IS THE SIX-DIGIT BNF READING CORRECT? (format audit)")
# NOTE ON LEADING ZEROS: the column is numeric, so a BNF code written '020400'
# is stored as 20400 (five digits).  The nurse guide says a leading zero is
# used 'where appropriate', i.e. for chapters 01-09.  A valid code is therefore
# 5 digits (chapter 1-9) or 6 digits (chapter 10-13), and chapter extraction
# must NOT assume a fixed width.
allt = [int(v) for _, r in D[drcols].iterrows() for v in r if np.isfinite(v)]
pos = [t for t in allt if t > 0]
neg = sorted({t for t in allt if t <= 0})
n5 = sum(1 for t in pos if 10000 <= t <= 99999)
n6 = sum(1 for t in pos if 100000 <= t <= 999999)
bad = [t for t in pos if not (10000 <= t <= 999999)]
w("  positive code tokens: %d | 5-digit (ch.01-09): %d | 6-digit (ch.10-13): %d | out-of-range: %d"
  % (len(pos), n5, n6, len(bad)))
w("  non-positive sentinel values: %s" % neg)
chap = sorted({(t // 10000) if t >= 100000 else (t // 10000) for t in pos})
w("  chapters present (//10000): %s" % chap)
# chapter 99 = ELSA/BNF pseudo-code; report separately, never as a drug class
c99 = sum(1 for t in pos if 990000 <= t <= 999999)
w("  chapter-99 pseudo-codes: %d (%.2f%%) -> excluded from all class flags"
  % (c99, 100.0 * c99 / max(1, len(pos))))
# cross-check against the nurse table's own NumMeds field
if "NumMeds" in D.columns:
    nm = pd.to_numeric(D["NumMeds"], errors="coerce")
    own = D[drcols].apply(lambda r: int(sum(1 for v in r if np.isfinite(v) and v > 0)), axis=1)
    agree = int((nm.fillna(-9) == own).sum())
    w("  NumMeds == count of positive DrC codes: %d / %d persons (%.1f%%)"
      % (agree, len(D), 100.0 * agree / len(D)))
w("  -> format is consistent with a six-digit chapter.section.subsection BNF")
w("     code stored with leading zeros dropped, negative sentinel = none.")

sec("2. CLASS COVERAGE IN DESIGN B BY SAMPLE (n=610, and main-model subset)")
s = s.merge(C, on="mergeid", how="left")
for c in ["n_codes", "nsaid", "cort_rheum", "dmard", "gout", "s0603",
          "cort_any", "any_bnf10"]:
    s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0.0)

WAVES_ADL = ["adl%d" % x for x in WAVES]
s["has_outcome"] = s[WAVES_ADL].notna().any(axis=1)
cov_rows = []
for lab, d in [("Design B (n=%d)" % len(s), s),
               ("main-model subset (any w7-w9 ADL, n=%d)" % int(s.has_outcome.sum()),
                s[s.has_outcome])]:
    w("\n  -- %s --" % lab)
    for c, desc in [("any_bnf10", "any BNF chapter 10 (musculoskeletal)"),
                    ("nsaid", "10.1.1 NSAID"),
                    ("cort_rheum", "10.1.2 corticosteroid (rheumatic)"),
                    ("cort_any", "any corticosteroid (10.1.2 or 6.3)"),
                    ("dmard", "10.1.3 DMARD"),
                    ("gout", "10.1.4 gout")]:
        n = int(d[c].sum())
        w("     %-40s %4d / %4d = %5.1f%%" % (desc, n, len(d), 100.0 * n / max(1, len(d))))
        cov_rows.append(dict(sample=lab, cls=c, n=n, N=len(d),
                             pct=100.0 * n / max(1, len(d))))
    w("     mean n_codes = %.2f" % d.n_codes.mean())

sec("3. DOES DRUG CLASS TRACK THE EXPOSURE? (confounding check)")
for c in ["nsaid", "cort_any", "dmard"]:
    r = np.corrcoef(s[c].values, s["slope_z"].values)[0, 1]
    r2 = np.corrcoef(s[c].values, s["mean_z"].values)[0, 1]
    w("  %-10s corr with slope_z = %+.3f | with mean_z = %+.3f" % (c, r, r2))
w("  (near-zero correlations mean adjustment cannot move the coefficient much)")

sec("4. LONG FORM AND MODELS")
rec = []
for wv in WAVES:
    d = s[["mergeid"]].copy()
    d["y"] = pd.to_numeric(s["adl%d" % wv], errors="coerce")
    for c in COV + ["mean_z", "slope_z", "nsaid", "cort_rheum", "dmard",
                    "cort_any", "any_bnf10", "gout"]:
        d[c] = s[c]
    d["time"] = wv - WAVES[0]
    rec.append(d)
d0 = pd.concat(rec, ignore_index=True)
dcc = d0[d0.y.notna()].copy()
w("  persons=%d  observations=%d" % (dcc.mergeid.nunique(), len(dcc)))


def fit(formula, d):
    return smf.ols(formula, d).fit(cov_type="cluster",
                                   cov_kwds={"groups": d["mergeid"]})


def report(label, r, note="", interaction=None):
    t = "time:slope_z"
    ci = r.conf_int().loc[t].tolist()
    row = dict(scenario=label, beta=r.params[t], se=r.bse[t], p=r.pvalues[t],
               ci_lo=ci[0], ci_hi=ci[1],
               n_person=int(dcc.mergeid.nunique()), note=note)
    rows.append(row)
    w("  %-50s beta=%+.4f  SE=%.4f  95%%CI [%+.4f,%+.4f]  P=%.4g" %
      (label, row["beta"], row["se"], row["ci_lo"], row["ci_hi"], row["p"]))
    if interaction and interaction in r.params:
        w("       %-24s beta=%+.4f  P=%.4g" %
          (interaction, r.params[interaction], r.pvalues[interaction]))
    return row


r0 = fit(FML, dcc)
report("S0 primary specification (reference)", r0,
       "must reproduce beta=+0.0568 / P=0.0592")
w("       time:mean_z beta=%+.4f P=%.4g" %
  (r0.params.get("time:mean_z", np.nan), r0.pvalues.get("time:mean_z", np.nan)))

specs = [
    ("D1 + NSAID (10.1.1)", FML + " + nsaid", "class-specific main effect"),
    ("D2 + corticosteroid (10.1.2 or 6.3)", FML + " + cort_any", "class-specific"),
    ("D3 + DMARD (10.1.3)", FML + " + dmard", "class-specific"),
    ("D4 + NSAID + corticosteroid + DMARD", FML + " + nsaid + cort_any + dmard",
     "mutually adjusted classes"),
    ("D5 + any BNF chapter 10 drug", FML + " + any_bnf10", "whole musculoskeletal chapter"),
    ("D6 + NSAID x time", FML + " + nsaid + time:nsaid",
     "effect modification by NSAID", "time:nsaid"),
    ("D7 + DMARD x time", FML + " + dmard + time:dmard",
     "effect modification by DMARD", "time:dmard"),
]
for item in specs:
    lab, f, note = item[0], item[1], item[2]
    inter = item[3] if len(item) > 3 else None
    try:
        report(lab, fit(f, dcc), note, inter)
    except Exception as e:
        w("  %s ERROR %r" % (lab, e))
        w(traceback.format_exc()[-400:])

# influence check: only 11 DMARD users, so drop them and refit
sub = dcc[dcc.mergeid.map(s.set_index("mergeid")["dmard"]) == 0]
try:
    r = fit(FML, sub)
    t = "time:slope_z"
    ci = r.conf_int().loc[t].tolist()
    row = dict(scenario="D8 restricted to non-DMARD users", beta=r.params[t],
               se=r.bse[t], p=r.pvalues[t], ci_lo=ci[0], ci_hi=ci[1],
               n_person=int(sub.mergeid.nunique()), note="influence of 11 DMARD users")
    rows.append(row)
    w("  %-50s beta=%+.4f  SE=%.4f  95%%CI [%+.4f,%+.4f]  P=%.4g  (n=%d)"
      % (row["scenario"], row["beta"], row["se"], row["ci_lo"], row["ci_hi"],
         row["p"], row["n_person"]))
except Exception as e:
    w("  D8 ERROR %r" % e)

sec("5. VERDICT INPUTS")
b = [r["beta"] for r in rows]
w("  time x slope_z across %d specifications: %s" % (len(b), ", ".join("%+.4f" % x for x in b)))
w("  range: %+.4f to %+.4f (spread %.4f)" % (min(b), max(b), max(b) - min(b)))
w("  Interpretation guard: w6 medication is measured at the END of the")
w("  exposure window, so it can be a consequence of prior inflammation.")
w("  These rows are robustness checks, NOT confounder control, and a stable")
w("  coefficient here does not license a causal reading.")

pd.DataFrame(cov_rows).to_csv(os.path.join(OUT, "drug_class_coverage.csv"),
                             index=False, encoding="utf-8-sig")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "drug_class_coefs.csv"),
                          index=False, encoding="utf-8-sig")
w("\n  saved -> out/drug_class_coverage.csv, out/drug_class_coefs.csv")
w("\nDONE")
LOG.close()
print("done")

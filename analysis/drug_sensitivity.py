# -*- coding: utf-8 -*-
"""Reviewer round-3 item 6: w6 medication as a CONCURRENT covariate.

READ-ONLY with respect to every frozen result: the primary model is only
RE-FITTED as a reference row (same specification, same coefficients), never
rewritten on disk.  Writes out/DRUG_SENS.txt and out/drug_sens_coefs.csv.

IMPORTANT SCOPE: local catalogues contain NO code->drug-name dictionary and
the DrC code decimal position is undocumented (see REVIEW_AUDIT_R3B.txt).
Therefore NO NSAID / glucocorticoid / DMARD / biologic class mapping is
attempted.  What CAN be done without a dictionary is an overall medication
BURDEN indicator derived from the raw w6 nurse table:
   n_drugs   = number of positive DrC codes (5-6 digit), per person
   any_drug  = 1 if n_drugs > 0
   NumMeds   = the nurse table's own 'DRUG: NumMeds' field
   MedCNJD   = 'are they taking any medicines' self-report
These are NOT drug-class adjustments; they only test whether concurrent
medication burden confounds the time x CRP-slope coefficient.
"""
import os
import re
import traceback
import psycopg2
import numpy as np
import pandas as pd
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
LOG = open(os.path.join(OUT, "DRUG_SENS.txt"), "w", encoding="utf-8")


def w(*a):
    print(*a, file=LOG)
    LOG.flush()


def sec(t):
    w("\n" + "=" * 92)
    w("## " + t)
    w("=" * 92)


COV = ["age_w2", "female", "educl", "adl6", "bmi_w2", "smoken_w2", "n_comorb"]
FML = "y ~ time * mean_z + time * slope_z + " + " + ".join(COV)
rows = []


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

# ---------------------------------------------------------------- w6 drugs
CR = _db("elsa_raw")
cc = pd.read_sql(
    "SELECT column_name FROM information_schema.columns "
    "WHERE table_schema='public' AND table_name='rw_w6nurse_p1' "
    "ORDER BY ordinal_position", CR).column_name.tolist()
drcols = [c for c in cc if re.fullmatch(r"DrC\d+", c, re.I)]
extra = [c for c in ["NumMeds", "MedCNJD", "MedIntro"] if c in cc]
w("w6 nurse table: %d DrC columns ; extra fields %s" % (len(drcols), extra))
raw = pd.read_sql("SELECT mergeid, %s FROM rw_w6nurse_p1" %
                  ", ".join('"%s"' % c for c in drcols + extra), CR)
raw = raw.loc[:, ~raw.columns.duplicated()].copy()
raw["mergeid"] = raw["mergeid"].astype(str).str.strip()
D = raw[raw.mergeid.isin(ids)].copy()

sec("1. WHAT THE w6 DRUG FIELDS CONTAIN (Design B, n=%d)" % len(ids))
w("  Design B rows found in rw_w6nurse_p1: %d" % len(D))
for c in drcols:
    D[c] = pd.to_numeric(D[c], errors="coerce")
mat = D[drcols]
positive = mat.where(mat > 0)
# special/undocumented large codes
special = positive.where(positive >= 999990)
n_special = int(special.notna().sum().sum())
w("  positive DrC cells=%d ; cells >=999990 (undocumented special)=%d" %
  (int(positive.notna().sum().sum()), n_special))
D["n_drugs"] = positive.notna().sum(axis=1)
D["any_drug"] = (D.n_drugs > 0).astype(float)
w("  n_drugs: mean=%.3f  median=%.1f  max=%d ; any_drug=%d/%d (%.1f%%)" %
  (D.n_drugs.mean(), D.n_drugs.median(), D.n_drugs.max(),
   int(D.any_drug.sum()), len(D), 100 * D.any_drug.mean()))
for c in extra:
    v = pd.to_numeric(D[c], errors="coerce")
    w("  %-10s nonnull=%d  distribution=%s" %
      (c, int(v.notna().sum()), v.value_counts(dropna=False).head(8).to_dict()))

w("\n  NOTE on dictionary: elsa_raw_var_labels labels DrC* only as")
w("  'DRUG: Drug Code'; the 49 BNF-labelled variables are all HSE wave-0")
w("  fields (medbi*, incode*, medtypo*, pillbrd), NOT the w6 nurse DrC*.")
w("  No code->name lookup table exists locally and the decimal position of")
w("  the 5-6 digit codes is undocumented -> class mapping NOT attempted.")

# merge onto analysis frame
for c in ["n_drugs", "any_drug"] + extra:
    if c in D.columns:
        s[c] = s.mergeid.map(D.set_index("mergeid")[c])
s["n_drugs"] = s.n_drugs.fillna(0.0)
s["any_drug"] = s.any_drug.fillna(0.0)
s["n_drugs_z"] = z(s.n_drugs)

# ---------------------------------------------------------------- long form
WAVES = [7, 8, 9]
rec = []
for wv in WAVES:
    d = s[["mergeid"]].copy()
    d["y"] = pd.to_numeric(s["adl%d" % wv], errors="coerce")
    for c in COV + ["mean_z", "slope_z", "base_z", "n_drugs_z", "any_drug",
                    "n_drugs"]:
        d[c] = s[c]
    d["time"] = wv - WAVES[0]
    rec.append(d)
d0 = pd.concat(rec, ignore_index=True)
dcc = d0[d0.y.notna()].copy()


def fit(formula, d):
    return smf.ols(formula, d).fit(cov_type="cluster",
                                   cov_kwds={"groups": d["mergeid"]})


def report(label, r, note=""):
    t = "time:slope_z"
    ci = r.conf_int().loc[t].tolist()
    rec = dict(scenario=label, beta=r.params[t], se=r.bse[t], p=r.pvalues[t],
               ci_lo=ci[0], ci_hi=ci[1], n_person=int(dcc.mergeid.nunique()),
               note=note)
    rows.append(rec)
    w("  %-52s beta=%+.4f  SE=%.4f  95%%CI [%+.4f,%+.4f]  P=%.4g" %
      (label, rec["beta"], rec["se"], rec["ci_lo"], rec["ci_hi"], rec["p"]))
    return rec


sec("2. REFERENCE AND MEDICATION-BURDEN SENSITIVITY (complete-case, w7-w9 ADL)")
r0 = fit(FML, dcc)
report("S0 primary specification (reference)", r0,
       "should reproduce beta=+0.0568 / P=0.05924")
w("    n=%d persons / %d observations" % (dcc.mergeid.nunique(), len(dcc)))
w("    time:mean_z beta=%+.4f P=%.4g" %
  (r0.params.get("time:mean_z", np.nan), r0.pvalues.get("time:mean_z", np.nan)))

specs = [
    ("S1 + any_drug main effect", FML + " + any_drug", "binary concurrent medication"),
    ("S2 + any_drug x time", FML + " + any_drug + time:any_drug",
     "allows medication to affect ADL rate"),
    ("S3 + n_drugs_z x time", FML + " + n_drugs_z + time:n_drugs_z",
     "dose-like medication burden"),
]
for lab, f, note in specs:
    try:
        r = fit(f, dcc)
        report(lab, r, note)
        for t in ["time:any_drug", "time:n_drugs_z"]:
            if t in r.params:
                w("       %-20s beta=%+.4f P=%.4g" % (t, r.params[t], r.pvalues[t]))
    except Exception as e:
        w("  %s ERROR %r" % (lab, e))
        w(traceback.format_exc()[-500:])

# also: adjust for self-reported 'taking any medicines' if usable
if "MedCNJD" in dcc.columns:
    v = pd.to_numeric(dcc.MedCNJD, errors="coerce")
    if v.notna().sum() > 100:
        dcc["med_any"] = (v > 0).astype(float)
        try:
            r = fit(FML + " + med_any + time:med_any", dcc)
            report("S4 + self-reported any-medicine x time", r, "MedCNJD")
        except Exception as e:
            w("  S4 ERROR %r" % e)

sec("3. VERDICT INPUTS")
betas = [r["beta"] for r in rows]
w("  time x slope_z across specifications: %s" %
  ", ".join("%+.4f" % b for b in betas))
w("  These are OVERALL medication-burden adjustments.  They do NOT replace")
w("  class-specific NSAID / glucocorticoid / DMARD / biologic adjustment,")
w("  which requires an external drug dictionary that is not in this extract.")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "drug_sens_coefs.csv"),
                          index=False, encoding="utf-8-sig")
w("\n  saved -> out/drug_sens_coefs.csv")
w("\nDONE")
LOG.close()
print("done")

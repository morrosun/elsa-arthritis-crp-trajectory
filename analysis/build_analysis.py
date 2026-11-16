# -*- coding: utf-8 -*-
"""Build the analysis dataset shared by all three competing designs.

Outputs (all under out/):
  exposure_long.csv   cleaned CRP/FIB long table with t index, 4 time points
  elsa_base.csv       one row per person: covariates + all ADL/IADL outcomes
  cohort_designA/B/C.csv   the three competing cohorts
  charls_base.csv     CHARLS replication person-wave table
  BUILD_LOG.txt       audit
"""
import os
import traceback
import psycopg2
import pandas as pd
import numpy as np
import pyreadr


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
RDATA = os.environ.get("ELSA_RDATA",
                        os.path.join(BASE, "data", "AGING_DB_R"))
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "BUILD_LOG.txt"), "w", encoding="utf-8")


def w(*a):
    print(*a, file=LOG)
    LOG.flush()


def sec(t):
    w("\n" + "=" * 90)
    w("## " + t)
    w("=" * 90)


CE = _db("elsa")

# ---------------------------------------------------------------- 1. exposure long
sec("1. Exposure long table (4 draws: 2004 / 2008 / 2012 / 2016-19)")
crp = pd.read_csv(os.path.join(OUT, "crp_long.csv"), dtype={"mergeid": str})
crp["mergeid"] = crp.mergeid.astype(str).str.strip()
w("raw rows=%d persons=%d ; wave_raw values=%s" %
  (len(crp), crp.mergeid.nunique(), sorted(crp.wave_raw.dropna().unique())))
T_MAP = {2: 0.0, 4: 1.0, 6: 2.0, 89: 3.0}
crp["t"] = crp.wave_raw.map(T_MAP)
crp["logcrp"] = np.where(crp.crp_valid, np.log(crp.crp), np.nan)
crp["logfib"] = np.where(crp.fib_valid, np.log(crp.fib), np.nan)
crp["crp_no10"] = np.where(crp.crp_valid_no10, crp.crp, np.nan)
crp.to_csv(os.path.join(OUT, "exposure_long.csv"), index=False, encoding="utf-8-sig")
w("valid CRP rows=%d ; valid FIB rows=%d ; logcrp range %.2f..%.2f" %
  (crp.crp_valid.sum(), crp.fib_valid.sum(),
   np.nanmin(crp.logcrp), np.nanmax(crp.logcrp)))

# ---------------------------------------------------------------- 2. covariates (PG)
sec("2. ELSA covariates from harmonized PG shards")


def cols_of(tbl):
    d = pd.read_sql("""SELECT column_name FROM information_schema.columns
                       WHERE table_schema='public' AND table_name='%s'""" % tbl, CE)
    return set(d.column_name.str.lower().tolist())


want = {
    "gh_elsa_p1": ["r2agey", "ragender", "rabyear", "r2iwstat"],
    "gh_elsa_p2": ["raeducl", "raedyrs_e", "r2shlt"],
    "gh_elsa_p3": ["r2adla", "r2iadla", "r2adlam"],
    "gh_elsa_p4": ["r2iadla", "r2iadlfour"],
    "gh_elsa_p5": ["r2hibpe", "r2diabe", "r2hearte", "r2stroke", "r2lunge",
                   "r2cancre", "r2arthre", "r2bmi"],
}
sel = {}
for tbl, cs in want.items():
    avail = cols_of(tbl)
    hit = [c for c in cs if c.lower() in avail]
    miss = [c for c in cs if c.lower() not in avail]
    sel[tbl] = hit
    w("%s: using %s ; missing %s" % (tbl, hit, miss))

parts = []
for tbl, cs in sel.items():
    if not cs:
        continue
    expr = ", ".join("%s.%s AS %s" % (tbl, c, c) for c in cs)
    q = "SELECT %s.mergeid, %s FROM %s" % (tbl, expr, tbl)
    d = pd.read_sql(q, CE)
    d["mergeid"] = d.mergeid.astype(str).str.strip()
    parts.append(d)
    w("  %s -> %d rows x %d cols" % (tbl, len(d), len(d.columns)))

base = parts[0]
for d in parts[1:]:
    base = base.merge(d, on="mergeid", how="outer")
w("merged PG covariates: %d persons x %d cols" % base.shape)

# wide outcomes (arthritis, ADL/IADL w6-w9, iwstat)
wide = pd.read_csv(os.path.join(OUT, "analysis_wide.csv"), dtype={"mergeid": str})
wide["mergeid"] = wide.mergeid.astype(str).str.strip()
bcols = ["mergeid"] + ["arth%d" % i for i in range(1, 10)] + \
        ["adl%d" % i for i in range(6, 10)] + ["iadl%d" % i for i in range(6, 10)] + \
        ["iw%d" % i for i in range(1, 10)] + ["radyear"]
base = base.merge(wide[bcols], on="mergeid", how="outer")
w("after wide merge: %d persons x %d cols" % base.shape)

# ---------------------------------------------------------------- 3. covariates (RData w2)
sec("3. ELSA baseline covariates from RData wave 2")
e = pyreadr.read_r(os.path.join(RDATA, "elsa.RData"))["elsa"]
e["mergeid"] = e.idauniqc.astype(str).str.strip()
e2 = e[e.wave == 2].copy()
rcols = [c for c in ["agey", "mbmi", "mbmicat", "mweight", "mheight",
                     "smoken", "smokev", "cesd", "radiagarthr",
                     "adltot6", "shlt", "raeducl", "raedyrs_e"] if c in e2.columns]
e2 = e2[["mergeid"] + rcols].rename(columns={c: "w2_" + c for c in rcols})
for c in e2.columns:
    if c != "mergeid":
        e2[c] = pd.to_numeric(e2[c], errors="coerce")
base = base.merge(e2, on="mergeid", how="outer")
w("RData w2 cols taken: %s" % rcols)
for c in e2.columns:
    if c != "mergeid":
        w("  %-20s nonnull=%5d mean=%8.3f" % (c, base[c].notna().sum(), base[c].mean()))

# harmonize covariate coding
base["female"] = np.where(base.ragender == 2, 1.0, np.where(base.ragender == 1, 0.0, np.nan))
base["age_w2"] = pd.to_numeric(base.get("r2agey", base.get("w2_agey")), errors="coerce")
if "w2_agey" in base:
    base["age_w2"] = base["age_w2"].fillna(base["w2_agey"])
base["educl"] = pd.to_numeric(base.get("raeducl", base.get("w2_raeducl")), errors="coerce")
if "w2_raeducl" in base:
    base["educl"] = base["educl"].fillna(base["w2_raeducl"])
base["bmi_w2"] = pd.to_numeric(base.get("w2_mbmi"), errors="coerce")
base["smoken_w2"] = pd.to_numeric(base.get("w2_smoken"), errors="coerce")
base["cesd_w2"] = pd.to_numeric(base.get("w2_cesd"), errors="coerce")
base["adl_w2"] = pd.to_numeric(base.get("r2adla"), errors="coerce")
if "w2_adltot6" in base:
    base["adl_w2"] = base["adl_w2"].fillna(pd.to_numeric(base["w2_adltot6"], errors="coerce"))
base["arth_age"] = pd.to_numeric(base.get("w2_radiagarthr"), errors="coerce")
comorb = [c for c in ["r2hibpe", "r2diabe", "r2hearte", "r2stroke", "r2lunge"] if c in base]
for c in comorb:
    base[c] = pd.to_numeric(base[c], errors="coerce").where(lambda s: s.isin([0, 1]))
base["n_comorb"] = base[comorb].sum(axis=1, min_count=1) if comorb else np.nan
w("\nfinal covariate availability:")
for c in ["age_w2", "female", "educl", "bmi_w2", "smoken_w2", "cesd_w2",
          "adl_w2", "n_comorb", "arth_age"]:
    if c in base:
        w("  %-14s nonnull=%5d mean=%8.3f" % (c, base[c].notna().sum(), base[c].mean()))

# ---------------------------------------------------------------- 4. cohorts
sec("4. Three competing designs")
v = crp[crp.crp_valid]
nw = v.groupby("mergeid").size()
pid3 = set(nw[nw >= 3].index.astype(str))
pid4 = set(nw[nw == 4].index.astype(str))
v246 = v[v.wave_raw.isin([2, 4, 6])]
c246 = v246.groupby("mergeid").size()
pid246 = set(c246[c246 == 3].index.astype(str))
arth2 = set(base.loc[base.arth2 == 1, "mergeid"].astype(str))
allids = set(base.mergeid.astype(str))

designs = {
    "A": ("CORE / original: >=3 draws (incl. 2016-19) -> w6-w9 ADL slope",
          arth2 & pid3 & allids),
    "B": ("CLEAN: w2+w4+w6 (2004-12) -> w7-w9 ADL slope",
          arth2 & pid246 & allids),
    "C": ("STRICT: all 4 draws (2004-19) -> w9 single-time ADL",
          arth2 & pid4 & allids),
}
for k, (desc, ids) in designs.items():
    pd.DataFrame({"mergeid": sorted(ids)}).to_csv(
        os.path.join(OUT, "cohort_design%s.csv" % k), index=False)
    s = base[base.mergeid.isin(ids)]
    w("\nDesign %s: n=%d  | %s" % (k, len(ids), desc))
    w("  age_w2 %.1f (SD %.1f) ; female %.1f%% ; ADL w6 %.3f -> w9 %.3f" %
      (s.age_w2.mean(), s.age_w2.std(), 100 * s.female.mean(),
       s.adl6.mean(), s.adl9.mean()))
    w("  >=2 ADL waves w7-w9 : %d ; w9 ADL available : %d" %
      ((s[["adl7", "adl8", "adl9"]].notna().sum(axis=1) >= 2).sum(),
       s.adl9.notna().sum()))
base.to_csv(os.path.join(OUT, "elsa_base.csv"), index=False, encoding="utf-8-sig")
w("\nelsa_base.csv : %d persons x %d cols" % base.shape)

# ---------------------------------------------------------------- 5. CHARLS base
sec("5. CHARLS replication base")
ch = pyreadr.read_r(os.path.join(RDATA, "charls.RData"))["charls"]
ch["pid"] = ch.ID.astype(str).str.strip()
ccols = [c for c in ["wave", "arthre", "adlab_c", "iadl", "ragender", "raeducl",
                     "bmi", "smoken", "rabyear", "hibpe", "diabe", "hearte"] if c in ch]
ch = ch[["pid"] + ccols].copy()
for c in ccols:
    ch[c] = pd.to_numeric(ch[c], errors="coerce")
w("CHARLS panel rows=%d persons=%d ; cols=%s" % (len(ch), ch.pid.nunique(), ccols))
two = pd.read_csv(os.path.join(OUT, "charls_two_wave_crp.csv"), dtype={"pid": str})
two["pid"] = two.pid.astype(str).str.strip()
two["dcrp"] = two.crp_2015 - two.crp_2011
w("CHARLS two-wave CRP rows=%d ; dcrp mean %.3f SD %.3f" %
  (len(two), two.dcrp.mean(), two.dcrp.std()))
arth1 = set(ch.loc[(ch.wave == 1) & (ch.arthre == 1), "pid"].astype(str))
rep = arth1 & set(two.pid)
w("replication cohort (arthritis@2011 x both valid CRP): %d" % len(rep))
cs = ch[ch.pid.isin(rep)].copy()
cs["age2011"] = 2011 - cs.rabyear
cs = cs.merge(two[["pid", "crp_2011", "crp_2015", "dcrp"]], on="pid", how="left")
cs.to_csv(os.path.join(OUT, "charls_base.csv"), index=False, encoding="utf-8-sig")
w("CHARLS outcome availability by wave (replication cohort): %s" %
  cs.groupby("wave").agg(n_adl=("adlab_c", "count"), n_iadl=("iadl", "count")).to_dict())
w("charls_base.csv : %d rows" % len(cs))
LOG.close()
print("done")

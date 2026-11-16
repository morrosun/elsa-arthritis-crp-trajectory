# -*- coding: utf-8 -*-
"""Direction 3 feasibility audit, part 2: core intersection, outcomes,
   temporal-overlap alternatives, CHARLS replication.
   Writes out/feasibility_02.txt + out/analysis_wide.csv + cohort CSVs."""
import os
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
LOG = open(os.path.join(OUT, "feasibility_02.txt"), "w", encoding="utf-8")


def w(*a):
    print(*a, file=LOG)
    LOG.flush()


def sec(t):
    w("\n" + "=" * 92)
    w("## " + t)
    w("=" * 92)


CE = _db("elsa")
CC = _db("charls")

# ------------------------------------------------------------------
sec("1. ELSA analysis wide table (harmonized, PG)")
cols_arth = ", ".join("p5.r%darthre AS arth%d" % (i, i) for i in range(1, 10))
cols_adl = ", ".join("p3.r%dadla AS adl%d" % (i, i) for i in range(6, 10))
cols_iadl = ", ".join("p3.r%diadla AS iadl%d" % (i, i) for i in range(6, 10))
cols_iw = ", ".join("p1.r%diwstat AS iw%d" % (i, i) for i in range(1, 10))
sql = """SELECT p5.mergeid, %s, %s, %s, %s,
                k.radyear, k.ragender, k.rabyear
         FROM gh_elsa_p5 p5
         JOIN gh_elsa_p3 p3 USING (mergeid)
         JOIN gh_elsa_p1 p1 USING (mergeid)
         JOIN elsa_keys   k  USING (mergeid)""" % (
    cols_arth, cols_adl, cols_iadl, cols_iw)
wide = pd.read_sql(sql, CE)
w("wide table: %d rows x %d cols" % wide.shape)
wide["mergeid"] = wide["mergeid"].astype(str)
for c in wide.columns:
    if c != "mergeid":
        wide[c] = pd.to_numeric(wide[c], errors="coerce")
# Stata extended missing codes are ~2.147e9 -> mask out
for i in range(1, 10):
    wide["arth%d" % i] = wide["arth%d" % i].where(wide["arth%d" % i].isin([0, 1]))
for i in range(6, 10):
    wide["adl%d" % i] = wide["adl%d" % i].where(wide["adl%d" % i].between(0, 5))
    wide["iadl%d" % i] = wide["iadl%d" % i].where(wide["iadl%d" % i].between(0, 3))
for i in range(1, 10):
    wide["iw%d" % i] = wide["iw%d" % i].where(wide["iw%d" % i].between(0, 9))
wide["radyear"] = wide["radyear"].where(wide["radyear"].between(1900, 2025))
w("arthritis (valid 0/1) by wave: %s" %
  {i: (int((wide["arth%d" % i] == 1).sum()), int(wide["arth%d" % i].notna().sum()))
   for i in range(1, 10)})
w("ADL (0-5) non-missing w6-w9: %s" %
  {i: int(wide["adl%d" % i].notna().sum()) for i in range(6, 10)})
w("IADL (0-3) non-missing w6-w9: %s" %
  {i: int(wide["iadl%d" % i].notna().sum()) for i in range(6, 10)})
wide.to_csv(os.path.join(OUT, "analysis_wide.csv"), index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------
sec("2. CRP cleaned long table (from part 1)")
crp = pd.read_csv(os.path.join(OUT, "crp_long.csv"), dtype={"mergeid": str})
w("crp_long: %d rows / %d persons" % (len(crp), crp.mergeid.nunique()))
v = crp[crp.crp_valid]
nwave = v.groupby("mergeid").size()
w(">=3 valid CRP waves: %d persons ; =4: %d" % ((nwave >= 3).sum(), (nwave == 4).sum()))
pid3 = set(nwave[nwave >= 3].index.astype(str))
pid4 = set(nwave[nwave == 4].index.astype(str))
pid2 = set(nwave[nwave >= 2].index.astype(str))
# exposure restricted to w2/w4/w6
v246 = v[v.wave_raw.isin([2, 4, 6])]
n3_246 = v246.groupby("mergeid").size()
pid_w246_all3 = set(n3_246[n3_246 == 3].index.astype(str))
pid_w246_ge3 = set(n3_246[n3_246 >= 2].index.astype(str))
w("all 3 of w2/w4/w6 valid: %d persons" % len(pid_w246_all3))

all_ids = set(wide.mergeid.astype(str))

# ------------------------------------------------------------------
sec("3. CORE INTERSECTION (the go/no-go number)")
arth2 = set(wide.loc[wide.arth2 == 1, "mergeid"])
arth_ever26 = set(wide.loc[(wide[["arth%d" % i for i in range(2, 7)]] == 1).any(axis=1), "mergeid"])
arth_any = set(wide.loc[(wide[["arth%d" % i for i in range(1, 10)]] == 1).any(axis=1), "mergeid"])
w("arthritis @w2                       : %5d" % len(arth2))
w("arthritis ever, w2-w6               : %5d" % len(arth_ever26))
w("arthritis ever, any wave            : %5d" % len(arth_any))
w("persons with >=3 valid CRP          : %5d" % len(pid3 & all_ids))
w("persons with 4 valid CRP            : %5d" % len(pid4 & all_ids))
w("")
w("=== CORE  : arthritis@w2  x  >=3 CRP : %5d ===" % len(arth2 & pid3 & all_ids))
w("=== STRICT: arthritis@w2  x   4 CRP  : %5d ===" % len(arth2 & pid4 & all_ids))
w("=== LOOSE : arthritis ever(w2-6) x >=3 CRP : %5d ===" %
  len(arth_ever26 & pid3 & all_ids))
w("=== ANY   : arthritis ever(any wave) x >=3 CRP : %5d ===" %
  len(arth_any & pid3 & all_ids))
w("=== WIDE  : arthritis@w2  x  >=2 CRP : %5d ===" % len(arth2 & pid2 & all_ids))
w("")
w("retention from arthritis@w2 to CORE: %.1f%%" %
  (100.0 * len(arth2 & pid3 & all_ids) / max(1, len(arth2))))

core = arth2 & pid3 & all_ids
strict = arth2 & pid4 & all_ids
loose = arth_ever26 & pid3 & all_ids

# ------------------------------------------------------------------
sec("4. Outcomes inside each candidate cohort")


def cohort_report(name, ids):
    s = wide[wide.mergeid.isin(ids)].copy()
    w("\n--- %s (n=%d) ---" % (name, len(ids)))
    w("ADL/IADL availability (non-missing) and means, w6-w9:")
    rows = []
    for i in range(6, 10):
        rows.append(dict(wave=i,
                         n_adl=wide["adl%d" % i].notna().sum() if False else s["adl%d" % i].notna().sum(),
                         mean_adl=round(s["adl%d" % i].mean(), 3) if s["adl%d" % i].notna().any() else np.nan,
                         n_iadl=s["iadl%d" % i].notna().sum(),
                         mean_iadl=round(s["iadl%d" % i].mean(), 3) if s["iadl%d" % i].notna().any() else np.nan))
    w(pd.DataFrame(rows).to_string(index=False))
    nadl = (s[["adl%d" % i for i in range(6, 10)]].notna().sum(axis=1))
    w("persons with >=2 ADL waves in w6-w9 : %d" % (nadl >= 2).sum())
    w("persons with >=3 ADL waves in w6-w9 : %d" % (nadl >= 3).sum())
    w("persons with  4 ADL waves in w6-w9 : %d" % (nadl == 4).sum())
    dead = set()
    for i in range(3, 10):
        dead |= set(s.loc[s["iw%d" % i].isin([5, 6]), "mergeid"])
    w("DEATHS (iwstat 5/6, w3-w9)         : %d persons (%.1f%%)" %
      (len(dead), 100.0 * len(dead) / max(1, len(ids))))
    dead_79 = set()
    for i in range(7, 10):
        dead_79 |= set(s.loc[s["iw%d" % i].isin([5, 6]), "mergeid"])
    w("deaths flagged w7-w9               : %d" % len(dead_79))
    w("baseline age @w2 (from panel)      : see below")
    return s, dead


s_core, dead_core = cohort_report("CORE: arth@w2 x >=3 CRP", core)
s_strict, dead_strict = cohort_report("STRICT: arth@w2 x 4 CRP", strict)
s_loose, dead_loose = cohort_report("LOOSE: arth ever w2-6 x >=3 CRP", loose)

# baseline age from RData panel
elsa = pyreadr.read_r(os.path.join(RDATA, "elsa.RData"))["elsa"]
elsa["mergeid"] = elsa["idauniqc"].astype(str)
b2 = elsa[(elsa.wave == 2) & elsa.mergeid.isin(core)]
w("\nCORE baseline @w2: n=%d, age mean=%.1f (SD %.1f), female=%.1f%%" %
  (len(b2), pd.to_numeric(b2.agey, errors="coerce").mean(),
   pd.to_numeric(b2.agey, errors="coerce").std(),
   100.0 * (pd.to_numeric(b2.ragender, errors="coerce") == 2).mean()))

# ------------------------------------------------------------------
sec("5. TEMPORAL OVERLAP: two design options")
w("Actual CRP time points       : 2004/05, 2008/09, 2012/13, 2016-19  (4 draws, 15y span)")
w("Proposed outcome w6-w9       : 2012, 2014, 2016, 2018")
w("=> w6 (2012) and w8/9 (2016-19) CRP are CONCURRENT/AFTER the outcome start.")
w("   A 4-point trajectory CANNOT precede a w6-w9 disability trajectory.")
w("")
w("OPTION B (clean temporality): exposure = w2/w4/w6 (2004/08/12);")
w("                              outcome = w7/w8/w9 (2014/16/18) ADL/IADL + death")
idsB = pid_w246_all3 & arth2 & all_ids
w("  persons with ALL w2,w4,w6 valid CRP : %d" % len(pid_w246_all3 & all_ids))
w("  OPTION B cohort (x arth@w2)         : %d" % len(idsB))
sB = wide[wide.mergeid.isin(idsB)]
nadlB = sB[["adl7", "adl8", "adl9"]].notna().sum(axis=1)
w("    >=2 ADL waves in w7-w9            : %d" % (nadlB >= 2).sum())
w("    >=3 ADL waves in w7-w9            : %d" % (nadlB >= 3).sum())
deadB = set()
for i in range(7, 10):
    deadB |= set(sB.loc[sB["iw%d" % i].isin([5, 6]), "mergeid"])
w("    deaths w7-w9                       : %d" % len(deadB))
w("")
w("OPTION C (max exposure, single-time outcome): all 4 CRP draws -> w9 ADL/death")
sC = wide[wide.mergeid.isin(strict)]
w("  STRICT cohort n=%d ; w9 ADL available: %d ; deaths w7-9: %d" %
  (len(strict), sC.adl9.notna().sum(), len(dead_strict)))

# ------------------------------------------------------------------
sec("6. CRP>10 exclusion and sentinel-code sensitivity (CORE definition)")
w("Definition                                             n_persons(>=3 waves)")
w("  A. negative sentinels removed, CRP>10 EXCLUDED        : %d" % len(pid3 & all_ids))
nv = crp[crp.crp > 0]
nw_no10 = nv.groupby("mergeid").size()
w("  B. negative sentinels removed, CRP>10 KEPT            : %d" %
  len(set(nw_no10[nw_no10 >= 3].index.astype(str)) & all_ids))
# only -1/-11 removed (naive rule)
naive = crp[~crp.crp.isin([-1.0, -11.0]) & crp.crp.notna()]
nw_naive = naive.groupby("mergeid").size()
w("  C. ONLY -1/-11 removed (WRONG: leaves -2/-3/-6/-7/-8): %d" %
  len(set(nw_naive[nw_naive >= 3].index.astype(str)) & all_ids))
w("     ^ this would leave %d negative values as 'real' CRP" %
  int(((naive.crp < 0)).sum()))
w("")
w("CORE n under each definition (x arth@w2):")
w("  A: %d   B: %d   C: %d" %
  (len(arth2 & pid3 & all_ids),
   len(arth2 & set(nw_no10[nw_no10 >= 3].index.astype(str)) & all_ids),
   len(arth2 & set(nw_naive[nw_naive >= 3].index.astype(str)) & all_ids)))

# ------------------------------------------------------------------
sec("7. CHARLS replication (2011 / 2015 CRP)")
c11 = pd.read_sql("SELECT id, newcrp AS crp2011 FROM w2011_blood", CC)
c15 = pd.read_sql("SELECT id, bl_crp AS crp2015 FROM w2015_blood", CC)
for d, col in [(c11, "crp2011"), (c15, "crp2015")]:
    d["id"] = d["id"].astype(str).str.strip()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    w("%s: %d rows / %d ids | non-null=%d valid=%d >10=%d <0=%d mean=%.2f" %
      (col, len(d), d.id.nunique(), d[col].notna().sum(),
       ((d[col] > 0) & (d[col] <= 10)).sum(), (d[col] > 10).sum(),
       (d[col] < 0).sum(), d.loc[(d[col] > 0) & (d[col] <= 10), col].mean()))
m = c11.merge(c15, on="id", how="inner")
v11 = (m.crp2011 > 0) & (m.crp2011 <= 10)
v15 = (m.crp2015 > 0) & (m.crp2015 <= 10)
w("both waves: %d ids ; both CRP valid: %d" % (len(m), (v11 & v15).sum()))
m.to_csv(os.path.join(OUT, "charls_crp.csv"), index=False, encoding="utf-8-sig")

ch = pyreadr.read_r(os.path.join(RDATA, "charls.RData"))["charls"]
ch["pid"] = ch["ID"].astype(str).str.strip()
w("\nCHARLS RData: %d rows / %d persons" % (len(ch), ch.pid.nunique()))
w("sample RData ID: %s ; blood id: %s" % (ch.pid.iloc[0], m.id.iloc[0]))
# ID alignment diagnostics
def align(blood_ids, panel_ids):
    b, p = set(blood_ids), set(panel_ids)
    tests = {
        "exact": lambda x: x,
        "drop_last_char": lambda x: x[:-1],
        "pad_last2": lambda x: x[:-1] + x[-1].zfill(2),
        "prefix11": lambda x: x[:11],
    }
    out = {}
    for k, fn in tests.items():
        out[k] = len({fn(x) for x in b} & p)
    return out
w("ID alignment (blood -> panel) match counts: %s" % align(m.id.tolist(), ch.pid.tolist()))

ch["arthre"] = pd.to_numeric(ch["arthre"], errors="coerce")
w("CHARLS arthritis (arthre=1) by wave: %s" %
  ch[ch.arthre == 1].groupby("wave").size().to_dict())
w("CHARLS ADL/IADL cols: %s" % [c for c in ch.columns if "adl" in c.lower()])
w("CHARLS iwstat by wave (code 5=died this wave): %s" %
  ch[ch.iwstat == 5].groupby("wave").size().to_dict()
  if "iwstat" in ch else "n/a")

# ------------------------------------------------------------------
sec("8. Outputs")
pd.DataFrame({"mergeid": sorted(core)}).to_csv(
    os.path.join(OUT, "cohort_core.csv"), index=False)
pd.DataFrame({"mergeid": sorted(strict)}).to_csv(
    os.path.join(OUT, "cohort_strict.csv"), index=False)
pd.DataFrame({"mergeid": sorted(idsB)}).to_csv(
    os.path.join(OUT, "cohort_optionB.csv"), index=False)
w("saved cohort_core.csv (%d), cohort_strict.csv (%d), cohort_optionB.csv (%d)" %
  (len(core), len(strict), len(idsB)))
LOG.close()
print("done")

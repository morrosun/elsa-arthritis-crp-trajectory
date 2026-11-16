# -*- coding: utf-8 -*-
"""Addendum: survivorship bias in death outcome; ELSA death data sources;
   CHARLS replication sample size after ID alignment."""
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
f = open(os.path.join(OUT, "ADDENDUM.txt"), "w", encoding="utf-8")


def w(*a):
    print(*a, file=f)
    f.flush()


def sec(t):
    w("\n" + "=" * 92)
    w("## " + t)
    w("=" * 92)


CE = _db("elsa")
CR = _db("elsa_raw")
CC = _db("charls")

core = pd.read_csv(os.path.join(OUT, "cohort_core.csv"), dtype={"mergeid": str})
core_ids = set(core.mergeid)
wide = pd.read_csv(os.path.join(OUT, "analysis_wide.csv"), dtype={"mergeid": str})
crp = pd.read_csv(os.path.join(OUT, "crp_long.csv"), dtype={"mergeid": str})

sec("1. Why death = 0 in the core cohort: survivorship bias")
iwcols = ", ".join("r%diwstat AS iw%d" % (i, i) for i in range(1, 10))
iw = pd.read_sql("SELECT p1.mergeid, %s, k.radyear FROM gh_elsa_p1 p1 "
                 "JOIN elsa_keys k USING (mergeid)" % iwcols, CE)
iw["mergeid"] = iw.mergeid.astype(str)
w("FULL ELSA sample: deaths flagged per wave (iwstat 5 = died this wave; 6 = died prev)")
for i in range(1, 10):
    w("  w%d: 5->%4d  6->%4d  9->%4d" %
      (i, (iw["iw%d" % i] == 5).sum(), (iw["iw%d" % i] == 6).sum(),
       (iw["iw%d" % i] == 9).sum()))
anydead = set()
for i in range(1, 10):
    anydead |= set(iw.loc[iw["iw%d" % i].isin([5, 6]), "mergeid"])
w("FULL sample: persons ever flagged dead (5/6) = %d / %d" % (len(anydead), len(iw)))
w("CORE: persons ever flagged dead = %d / %d" % (len(anydead & core_ids), len(core_ids)))
w("CORE iw9 distribution: %s" %
  iw[iw.mergeid.isin(core_ids)].iw9.value_counts(dropna=False).to_dict())
w("=> >=3 blood draws REQUIRES surviving (and attending nurse visits) to 2012+,")
w("   so anyone who died earlier is structurally excluded. Death as a secondary")
w("   outcome has ~0 events in this design. This is a hard design flaw, not a bug.")

sec("2. Is there a usable ELSA mortality source elsewhere?")
w("elsa_keys.radyear non-null: %d / %d" %
  (iw.radyear.notna().sum(), len(iw)))
w("elsa_raw EOL module columns (death-related):")
try:
    d = pd.read_sql("""SELECT column_name FROM information_schema.columns
                       WHERE table_schema='public' AND table_name='rw_eol_p1'
                         AND (column_name ILIKE '%death%' OR column_name ILIKE '%died%'
                              OR column_name ILIKE '%date%' OR column_name ILIKE '%dec%'
                              OR column_name ILIKE '%month%' OR column_name ILIKE '%year%')
                       ORDER BY ordinal_position""", CR)
    w(d.to_string(index=False))
    w("rw_eol_p1 rows: %s" % pd.read_sql(
        "SELECT count(*) n FROM rw_eol_p1", CR).to_string(index=False))
except Exception as ex:
    w("ERROR %r" % ex)
w("var_labels mentioning death in elsa_raw:")
try:
    d = pd.read_sql("""SELECT DISTINCT var, src, wave, label FROM elsa_raw_var_labels
                       WHERE label ILIKE '%died%' OR label ILIKE '%death%' OR label ILIKE '%deceased%'
                       ORDER BY var LIMIT 60""", CR)
    w(d.to_string(index=False))
except Exception as ex:
    w("ERROR %r" % ex)

sec("3. CHARLS replication (ID alignment solved)")
c11 = pd.read_sql("SELECT id::text AS id, newcrp FROM w2011_blood", CC)
c15 = pd.read_sql("SELECT id::text AS id, bl_crp FROM w2015_blood", CC)
c11["pid"] = c11.id.str.strip().str[:10] + c11.id.str.strip().str[10:].str.zfill(2)
c15["pid"] = c15.id.str.strip()
c11["crp"] = pd.to_numeric(c11.newcrp, errors="coerce")
c15["crp"] = pd.to_numeric(c15.bl_crp, errors="coerce")
w("2011 CRP: n=%d, valid(0<crp<=10)=%d, >10=%d, <0=%d, mean=%.2f" %
  (len(c11), ((c11.crp > 0) & (c11.crp <= 10)).sum(), (c11.crp > 10).sum(),
   (c11.crp < 0).sum(), c11.loc[(c11.crp > 0) & (c11.crp <= 10), "crp"].mean()))
w("2015 CRP: n=%d, valid=%d, >10=%d, <0=%d, mean=%.2f" %
  (len(c15), ((c15.crp > 0) & (c15.crp <= 10)).sum(), (c15.crp > 10).sum(),
   (c15.crp < 0).sum(), c15.loc[(c15.crp > 0) & (c15.crp <= 10), "crp"].mean()))
m = c11[["pid", "crp"]].merge(c15[["pid", "crp"]], on="pid",
                              how="inner", suffixes=("_2011", "_2015"))
w("persons with BOTH blood waves: %d" % len(m))
both = m[(m.crp_2011 > 0) & (m.crp_2011 <= 10) & (m.crp_2015 > 0) & (m.crp_2015 <= 10)]
w("persons with BOTH valid CRP: %d" % len(both))
w("  CRP change (2015-2011): mean=%.3f SD=%.3f" %
  ((both.crp_2015 - both.crp_2011).mean(), (both.crp_2015 - both.crp_2011).std()))
both.to_csv(os.path.join(OUT, "charls_two_wave_crp.csv"), index=False)

ch = pyreadr.read_r(os.path.join(RDATA, "charls.RData"))["charls"]
ch["pid"] = ch.ID.astype(str).str.strip()
ch["arthre"] = pd.to_numeric(ch.arthre, errors="coerce")
ch["iwstat"] = pd.to_numeric(ch.iwstat, errors="coerce")
w("\nCHARLS arthritis by wave (arthre=1): %s" %
  ch[ch.arthre == 1].groupby("wave").size().to_dict())
w("CHARLS iwstat value counts: %s" % ch.iwstat.value_counts(dropna=False).head(10).to_dict())
w("CHARLS ADL/IADL cols: %s" % [c for c in ch.columns if "adl" in c.lower()])

# arthritis at wave 1 (2011) or wave 3 (2015)
arth1 = set(ch.loc[(ch.wave == 1) & (ch.arthre == 1), "pid"])
arth_ever13 = set(ch.loc[(ch.wave.isin([1, 3])) & (ch.arthre == 1), "pid"])
w("\narthritis @2011 (w1): %d" % len(arth1))
w("arthritis ever w1/w3  : %d" % len(arth_ever13))
w("=== REPLICATION: arthritis@2011 x 2 valid CRP : %d ===" % len(arth1 & set(both.pid)))
w("=== REPLICATION: arthritis ever(w1/w3) x 2 valid CRP : %d ===" %
  len(arth_ever13 & set(both.pid)))
w("    (for reference, Ye 2025 used CHARLS n=3,413 with 87 deaths)")

# outcome availability
out_ch = ch[ch.pid.isin(arth1 & set(both.pid))]
w("\nCHARLS outcome availability in replication cohort:")
for c in [x for x in ["adlab_c", "iadl", "iwstat"] if x in ch.columns]:
    w("  %s by wave (non-null): %s" %
      (c, out_ch.assign(v=pd.to_numeric(out_ch[c], errors="coerce"))
       .groupby("wave").v.count().to_dict()))

sec("4. Sensitivity: does dropping CRP>10 change ELSA conclusions?")
v = crp[crp.crp_valid]
nw = v.groupby("mergeid").size()
w(">=3 waves with CRP>10 excluded: %d persons" % (nw >= 3).sum())
nv = crp[crp.crp > 0]
nw2 = nv.groupby("mergeid").size()
w(">=3 waves with CRP>10 kept    : %d persons" % (nw2 >= 3).sum())
w("CRP>10 excluded observation rows: %d / %d positives (%.1f%%)" %
  ((crp.crp > 10).sum(), (crp.crp > 0).sum(),
   100.0 * (crp.crp > 10).sum() / max(1, (crp.crp > 0).sum())))
f.close()
print("done")

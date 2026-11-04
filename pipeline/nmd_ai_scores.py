# Количественные NMD-скоры NMDetective-AI (UCSC gbdb hg38 nmdDetectAi.bb):
# per-вариантный join по (chrom,start,transcript,codon,stop) -> вероятность NMD.
# Затем: (1) абляция mt_hgb ± nmd_ai на репортерных сплитах (количественный отрицательный контроль);
#        (2) Destoop-мини-валидация: RT alone vs RT × escape_ai против органоидных OR.
# Выход: data/model_table_nmd_ai.parquet, results/metrics_nmd_ai.csv, results/destoop_composite.csv
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from bx.bbi.bigbed_file import BigBedFile
from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor

TAB = pd.read_parquet("data/model_table_nmd.parquet")
TAB["GENEINFO"] = TAB.GENEINFO.astype(str).replace("None", "NO_GENE")
bb = BigBedFile(open("data/ucsc/nmdDetectAi.bb", "rb"))

# ---------- 1) join скоров по координате варианта ----------
rows = []
for _, r in TAB.drop_duplicates("mutation_identifier").iterrows():
    chrom = str(r.Chromosome)
    if not chrom.startswith("chr"):
        chrom = "chr" + chrom
    try:
        hits = bb.get(chrom.encode(), int(r.start) - 1, int(r.start) + 2)
    except Exception:
        hits = []
    enst = str(r.Transcript_ID).split(".")[0]
    for h in hits or []:
        f = str(h).split("\t")
        # name: GENE:REFcodonNNN>STOP ; fields: chrom,start,end,name,score,strand,...,value,gene,transcript,codon,ref,stop
        if len(f) >= 14 and enst in f[11] and f[13] == r.stop_type.upper():
            rows.append(dict(mutation_identifier=r.mutation_identifier,
                             nmd_ai=float(f[10]), codon=f[12], ref=f[13]))
            break
sc = pd.DataFrame(rows)
print(f"присоединено NMDetective-AI скоров: {len(sc)} из {TAB.mutation_identifier.nunique()} вариантов", flush=True)
TAB = TAB.merge(sc[["mutation_identifier", "nmd_ai"]].drop_duplicates("mutation_identifier"),
                on="mutation_identifier", how="left")
TAB["escape_ai"] = 1 - TAB.nmd_ai      # больше скор -> сильнее NMD -> меньше escape
TAB.to_parquet("data/model_table_nmd_ai.parquet")

# ---------- 2) абляция (репортер): mt_hgb ± nmd_ai ----------
DRUGS = sorted([t for t in TAB.treatment.unique() if t != "Untreated"])
CATS = ["stop_type", "up_123nt", "down_123nt", "up_codon_2", "up_aa"]
NUMS = ["aa_gene_position", "nt_CDS_pos", "GC%", "deltaG_OA", "deltaG_mFold",
        "tAI_up_1aa", "CAI_up_1aa", "dist_to_downstreamstop"]

def prep(s, extra):
    X = s[CATS + NUMS + extra].copy()
    for c in NUMS + extra:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in CATS:
        X[c] = X[c].astype(str).fillna("NA").astype("category")
    return X

def mt_eval(tr, te, extra):
    Xtr, Xte = prep(tr, extra), prep(te, extra)
    Xtr["treatment"] = tr.treatment.astype(str).astype("category")
    Xte["treatment"] = pd.Categorical(te.treatment.astype(str), categories=Xtr.treatment.cat.categories)
    for c in CATS:
        Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                      categorical_features="from_dtype", random_state=0)
    m.fit(Xtr, tr.y)
    p = m.predict(Xte)
    return spearmanr(te.y, p).statistic

res = []
var_meta = TAB.drop_duplicates("mutation_identifier").set_index("mutation_identifier").GENEINFO
for extra, tag in [([], "mt_hgb"), (["escape_ai"], "mt_hgb+escapeAI")]:
    for drug in DRUGS:
        d = TAB[TAB.treatment == drug].dropna(subset=["y"])
        vids = d.mutation_identifier.unique()
        for tr_v, te_v in GroupKFold(5).split(vids, groups=pd.Series(vids).map(var_meta)):
            tr = TAB[TAB.treatment.isin(DRUGS) & TAB.mutation_identifier.isin(set(vids[tr_v]))].dropna(subset=["y"])
            te = d[d.mutation_identifier.isin(set(vids[te_v]))]
            res.append(dict(split="genegroup5", model=tag, drug=drug,
                            spearman=mt_eval(tr, te, extra)))
    tr = TAB[TAB.treatment.isin(DRUGS) & (TAB.GENEINFO != "CFTR")].dropna(subset=["y"])
    te = TAB[(TAB.treatment == drug) & (TAB.GENEINFO == "CFTR")]
    res.append(dict(split="cftr_holdout", model=tag, drug=drug,
                    spearman=mt_eval(tr, te, extra)))
    print(tag, "done", flush=True)

r = pd.DataFrame(res)
agg = r.groupby(["split", "model"]).spearman_mean_ = None
summ = r.groupby(["split", "model"]).spearman.mean().round(3)
print("\nабляция (репортер):\n", summ.to_string())
r.to_csv("results/metrics_nmd_ai.csv", index=False)

# ---------- 3) Destoop-мини-валидация с количественной осью ----------
or_destoop = {"G542X": 4.54, "S466X": 1.11, "Q525X": 1.17,
              "G550X": 1.0, "R553X": 1.0, "E585X": 1.0, "W1282X": 0.14}
cf = pd.read_csv("data/cftr_measured.csv", index_col=0)
drugs = [c for c in cf.columns if c != "Untreated"]
sub = TAB[(TAB.GENEINFO == "CFTR")].drop_duplicates("mutation_identifier_short").set_index("mutation_identifier_short")
d = pd.DataFrame({"OR": or_destoop})
d["RT_best"] = cf.loc[d.index, drugs].max(axis=1)
d["escape_ai"] = pd.to_numeric(sub.loc[d.index, "escape_ai"], errors="coerce")
d["composite"] = np.log10(d.RT_best) + np.log10(np.clip(d.escape_ai, 0.05, None))
d.round(3).to_csv("results/destoop_composite.csv")
print("\nDestoop (n=%d):" % len(d))
print(d.round(3).to_string())
for col in ["RT_best", "escape_ai", "composite"]:
    s = spearmanr(d.OR, d[col]); k = kendalltau(d.OR, d[col])
    print(f"  {col:12s} Spearman {s.statistic:+.3f} (p={s.pvalue:.3f})  Kendall {k.statistic:+.3f}")

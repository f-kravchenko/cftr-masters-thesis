# Этап 3: NMD-признаки из GENCODE v47 (механистические, как ядро правил NMDetective):
# dist_last_ejc — дистанция PTC до последнего экзон-экзонного соединения (нт, транскриптные коорд.),
# utr3_len — длина 3'UTR, exon_count, cds_fraction.
# Затем абляция H4 (репортерная часть): mt_hgb ± NMD-признаки на genegroup5 и cftr_holdout.
# Выход: data/model_table_nmd.parquet, results/metrics_nmd_ablation.csv
import warnings, gzip; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor

# ---------- 1) парсинг GTF: экзоны и CDS по транскриптам ----------
tx = {}   # enst_base -> dict(exons=[(s,e)], cds=[(s,e)], chrom, strand)
with gzip.open("data/gencode47.gtf.gz", "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.split("\t")
        feat, start, end, info = p[2], int(p[3]), int(p[4]), p[8]
        if feat not in ("exon", "CDS"):
            continue
        tid = info.split("transcript_id \"", 1)[1].split("\"", 1)[0]
        base = tid.split(".")[0]
        d = tx.setdefault(base, {"exons": [], "cds": []})
        d["exons" if feat == "exon" else "cds"].append((start, end))

print("транскриптов в GTF:", len(tx), flush=True)

def feats(d):
    ex = sorted(d["exons"]); cd = sorted(d["cds"])
    if not ex or not cd:
        return None
    lens = [e - s + 1 for s, e in ex]
    tx_len = sum(lens)
    # последняя экзон-экзонная шпилька = конец предпоследнего экзона (в транскриптных коорд.)
    last_ejc = tx_len - lens[-1]
    # CDS старт/конец в транскриптных координатах
    pos, cds_start, cds_end, acc = 0, None, None, 0
    for (s, e), L in zip(ex, lens):
        nxt = pos + L
        for cs, ce in cd:
            if s <= cs <= e and cds_start is None:
                cds_start = pos + (cs - s)
            if s <= ce <= e:
                cds_end = pos + (ce - s)
        pos = nxt
    if cds_start is None or cds_end is None:
        return None
    return dict(tx_len=tx_len, last_ejc=last_ejc, cds_start=cds_start,
                cds_len=cds_end - cds_start + 1, utr3_len=tx_len - cds_end,
                exon_count=len(ex))

rows = {k: feats(v) for k, v in tx.items()}
gdf = pd.DataFrame({k: v for k, v in rows.items() if v}).T.reset_index(names="enst")
gdf.to_parquet("data/gencode_nmd_feats.parquet")
print("с признаками:", len(gdf), flush=True)

# ---------- 2) join к model_table ----------
TAB = pd.read_parquet("data/model_table.parquet").dropna(subset=["RT"]).copy()
TAB["GENEINFO"] = TAB.GENEINFO.astype(str).replace("None", "NO_GENE")
TAB["y"] = np.log10(TAB.RT)
TAB["enst"] = TAB.Transcript_ID.astype(str).str.split(".").str[0]
before = len(TAB)
TAB = TAB.merge(gdf, on="enst", how="left")
print(f"присоединено NMD-признаков: {TAB.tx_len.notna().mean():.1%} строк (было {before})", flush=True)
TAB["nt_CDS_pos_f"] = pd.to_numeric(TAB.nt_CDS_pos, errors="coerce")
TAB["ptc_tx_pos"] = TAB.cds_start + TAB.nt_CDS_pos_f
TAB["dist_last_ejc"] = TAB.last_ejc - TAB.ptc_tx_pos        # >0: PTC выше последнего EJC (NMD-чувствительно)
TAB["cds_fraction"] = TAB.nt_CDS_pos_f / TAB.cds_len
NMD_FEATS = ["dist_last_ejc", "utr3_len", "exon_count", "cds_fraction"]
TAB.to_parquet("data/model_table_nmd.parquet")

# ---------- 3) абляция: mt_hgb ± NMD ----------
DRUGS = sorted([t for t in TAB.treatment.unique() if t != "Untreated"])
CATS = ["stop_type", "up_123nt", "down_123nt", "up_codon_2", "up_aa"]
NUMS = ["aa_gene_position", "nt_CDS_pos", "GC%", "deltaG_OA", "deltaG_mFold",
        "tAI_up_1aa", "CAI_up_1aa", "dist_to_downstreamstop"]

def prep(s, use_nmd):
    cols = CATS + NUMS + (NMD_FEATS if use_nmd else [])
    X = s[cols].copy()
    for c in NUMS + (NMD_FEATS if use_nmd else []):
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in CATS:
        X[c] = X[c].astype(str).fillna("NA").astype("category")
    return X

def mt_eval(tr, te, use_nmd):
    Xtr, Xte = prep(tr, use_nmd), prep(te, use_nmd)
    Xtr["treatment"] = tr.treatment.astype(str).astype("category")
    Xte["treatment"] = pd.Categorical(te.treatment.astype(str),
                                      categories=Xtr.treatment.cat.categories)
    for c in CATS:
        Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                      categorical_features="from_dtype", random_state=0)
    m.fit(Xtr, tr.y)
    p = m.predict(Xte)
    return spearmanr(te.y, p).statistic, float(np.sqrt(np.mean((te.y - p) ** 2)))

res = []
for use_nmd in [False, True]:
    tag = "mt_hgb" + ("+NMD" if use_nmd else "")
    # genegroup5 (фолды по вариантам, группы гены)
    var_meta = TAB.drop_duplicates("mutation_identifier").set_index("mutation_identifier").GENEINFO
    for drug in DRUGS:
        d = TAB[TAB.treatment == drug]
        vids = d.mutation_identifier.unique()
        for tr_v, te_v in GroupKFold(5).split(vids, groups=pd.Series(vids).map(var_meta)):
            tr = TAB[TAB.treatment.isin(DRUGS) & TAB.mutation_identifier.isin(set(vids[tr_v]))]
            te = d[d.mutation_identifier.isin(set(vids[te_v]))]
            rho, rmse = mt_eval(tr, te, use_nmd)
            res.append(dict(split="genegroup5", model=tag, drug=drug, n_test=len(te),
                            spearman=rho, rmse_log=rmse))
    for drug in DRUGS:  # cftr_holdout
        tr = TAB[TAB.treatment.isin(DRUGS) & (TAB.GENEINFO != "CFTR")]
        te = TAB[(TAB.treatment == drug) & (TAB.GENEINFO == "CFTR")]
        rho, rmse = mt_eval(tr, te, use_nmd)
        res.append(dict(split="cftr_holdout", model=tag, drug=drug, n_test=len(te),
                        spearman=rho, rmse_log=rmse))
    print(tag, "done", flush=True)

r = pd.DataFrame(res)
r.to_csv("results/metrics_nmd_ablation.csv", index=False)
s = r.groupby(["split", "model"]).agg(spearman_mean=("spearman", "mean"),
                                      rmse_log_mean=("rmse_log", "mean")).round(3)
s.to_csv("results/summary_nmd_ablation.csv")
print(s.to_string())

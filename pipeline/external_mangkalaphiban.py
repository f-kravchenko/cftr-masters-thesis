# Этап 3, внешняя валидация №1: Mangkalaphiban 2024 (13 CFTR PTC, dual-luciferase, G418).
# Модели (enet_tuned, mt_hgb) обучаем БЕЗ CFTR; предсказываем fold-change RT(G418)/RT(Untreated)
# и сравниваем: (а) с их измеренным fold-change (валидация модели),
# (б) межсистемное согласование двух экспериментов (их dual-luc vs наш sort-seq).
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

SEED = 0
TAB = pd.read_parquet("data/model_table.parquet").dropna(subset=["RT"]).copy()
TAB["GENEINFO"] = TAB.GENEINFO.astype(str).replace("None", "NO_GENE")
TAB["y"] = np.log10(TAB.RT)
DRUGS = sorted([t for t in TAB.treatment.unique() if t != "Untreated"])
CATS = ["stop_type", "up_123nt", "down_123nt", "up_codon_2", "up_aa"]
NUMS = ["aa_gene_position", "nt_CDS_pos", "GC%", "deltaG_OA", "deltaG_mFold",
        "tAI_up_1aa", "CAI_up_1aa", "dist_to_downstreamstop"]
ENET_FEATS = ["stop_type", "up_123nt", "down_123nt"]

def prep(s, with_drug, drug_tr=None):
    X = s[CATS + NUMS].copy()
    for c in NUMS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in CATS:
        X[c] = X[c].astype(str).fillna("NA").astype("category")
    return X

# их измерения
MK = "external/mangkalaphiban/Jacobson-Lab-AG_readthrough-3a69648/Figures/Data/CFTR_PTCs_DualLuc_measured_readthrough.csv"
mk = pd.read_csv(MK)
avg = mk.pivot_table(index="Mutation", columns="G418_treatment", values="Average")
mk["fold"] = mk.Mutation.map(avg["G418-treated"] / avg["Untreated"])
mk = mk.drop_duplicates("Mutation").set_index("Mutation")[["fold"]]

# наши toledano-измерения тех же вариантов (для межсистемного сравнения)
ours = TAB[(TAB.GENEINFO == "CFTR")].pivot_table(
    index="mutation_identifier_short", columns="treatment", values="RT")
mk = mk.join(ours[["G418", "Untreated"]].rename(
    columns={"G418": "our_G418", "Untreated": "our_Untreated"}))
mk["our_fold"] = mk.our_G418 / mk.our_Untreated

# обучение без CFTR и предсказание G418/Untreated для 13 вариантов
te = TAB[(TAB.GENEINFO == "CFTR") & TAB.mutation_identifier_short.isin(mk.index)]
preds = {}
for model in ["enet_tuned", "mt_hgb"]:
    ys = {}
    for drug in ["G418", "Untreated"]:
        tr = TAB[(TAB.treatment == drug) & (TAB.GENEINFO != "CFTR")]
        ted = te[te.treatment == drug]
        if model == "enet_tuned":
            pipe = make_pipeline(
                ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), ENET_FEATS)]),
                ElasticNet(alpha=1e-3, l1_ratio=0.1, max_iter=20000, random_state=SEED))
            pipe.fit(tr[ENET_FEATS], tr.y)
            ys[drug] = pd.Series(pipe.predict(ted[ENET_FEATS]).tolist(),
                                 index=ted.mutation_identifier_short)
        else:
            trm = TAB[TAB.treatment.isin(DRUGS + ["Untreated"]) & (TAB.GENEINFO != "CFTR")]
            Xtr, Xte = prep(trm, True), prep(ted, True)
            Xtr["treatment"] = trm.treatment.astype(str).astype("category")
            Xte["treatment"] = pd.Categorical(ted.treatment.astype(str),
                                              categories=Xtr.treatment.cat.categories)
            for c in CATS:
                Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)
            m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                              categorical_features="from_dtype", random_state=SEED)
            m.fit(Xtr, trm.y)
            ys[drug] = pd.Series(m.predict(Xte).tolist(), index=ted.mutation_identifier_short)
    preds[model] = 10 ** (ys["G418"] - ys["Untreated"])     # log10 -> fold

rows = []
def cmp(name, a, b):
    ok = a.notna() & b.notna()
    r = dict(comparison=name, n=int(ok.sum()),
             spearman=spearmanr(a[ok], b[ok]).statistic,
             pearson_log=pearsonr(np.log(a[ok]), np.log(b[ok])).statistic)
    rows.append(r)
    print(r, flush=True)

mk["enet_fold"] = preds["enet_tuned"]
mk["mt_fold"] = preds["mt_hgb"]
cmp("model_enet_fold_vs_Mangk_fold", mk.enet_fold, mk.fold)
cmp("model_mthgb_fold_vs_Mangk_fold", mk["mt_fold"], mk.fold)
cmp("experiment_Toledano_fold_vs_Mangk_fold", mk.our_fold, mk.fold)   # данные против данных

mk.round(3).to_csv("results/external_mangkalaphiban.csv")
pd.DataFrame(rows).round(3).to_csv("results/metrics_external_mangk.csv", index=False)
with open("results/summary_external_mangk.md", "w") as f:
    f.write("# Внешняя валидация №1: Mangkalaphiban (13 CFTR PTC, dual-luc, G418)\n\n")
    f.write(pd.DataFrame(rows).round(3).to_string(index=False))
    f.write("\n\nМодели обучены без CFTR (cftr_holdout-условия). Fold = RT(G418)/RT(Untreated) "
            "в каждой системе. Третья строка — согласование самих экспериментов "
            "(sort-seq HEK293T vs dual-luc HEK293), не моделей.\n\n")
    f.write(mk.round(3).to_string())
print("\nOK -> results/external_mangkalaphiban.csv")

# Этап 2: (1) elastic net с тюнингом (честный бейзлайн статьи);
# (2) drug-aware мультитаск HGB: препарат = категориальный признак на обычных сплитах,
#     few-shot LODO (H2): held-out препарат представлен 7-мерным профилем
#     Спирман-корреляций с остальными (калибровка на 300 вариантов, они исключены из теста).
# Выход: results/metrics_stage2.csv + results/summary_stage2.md
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

SEED = 0
TAB = pd.read_parquet("data/model_table.parquet").dropna(subset=["RT"]).copy()
TAB["GENEINFO"] = TAB.GENEINFO.astype(str).replace("None", "NO_GENE")
TAB["y"] = np.log10(TAB["RT"])
DRUGS = sorted([t for t in TAB.treatment.unique() if t != "Untreated"])

CATS = ["stop_type", "up_123nt", "down_123nt", "up_codon_2", "up_aa"]
NUMS = ["aa_gene_position", "nt_CDS_pos", "GC%", "deltaG_OA", "deltaG_mFold",
        "tAI_up_1aa", "CAI_up_1aa", "dist_to_downstreamstop"]
ENET_FEATS = ["stop_type", "up_123nt", "down_123nt"]
rows, best_hp = [], {}

def prep(s):
    X = s[CATS + NUMS].copy()
    for c in NUMS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in CATS:
        X[c] = X[c].astype(str).fillna("NA").astype("category")
    return X

def score(te, pred):
    m = ~np.isnan(pred)
    return (spearmanr(te.y[m], pred[m]).statistic,
            float(np.sqrt(np.mean((te.y[m] - pred[m]) ** 2))))

def hgb_fit_predict(tr, te, drug_col_tr, drug_col_te=None):
    """Мультитаск HGB: CATS+NUMS + treatment (категория). drug_col_* None -> без признака препарата."""
    Xtr, Xte = prep(tr), prep(te)
    if drug_col_tr is not None:
        Xtr["treatment"] = drug_col_tr.astype(str).astype("category")
        Xte["treatment"] = pd.Categorical(drug_col_te.astype(str),
                                          categories=Xtr.treatment.cat.categories)
    for c in CATS:
        Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                      categorical_features="from_dtype", random_state=SEED)
    m.fit(Xtr, tr.y)
    return m.predict(Xte)

# ---------- 1) elastic net с тюнингом ----------
def enet_tuned(tr, te, tag):
    pipe = make_pipeline(
        ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), ENET_FEATS)]),
        ElasticNet(max_iter=20000, random_state=SEED))
    gs = GridSearchCV(pipe,
        {"elasticnet__alpha": np.logspace(-4, 1, 12), "elasticnet__l1_ratio": [0.1, 0.5, 0.9]},
        cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(tr[ENET_FEATS], tr.y)
    best_hp[tag] = (round(gs.best_params_["elasticnet__alpha"], 5),
                    gs.best_params_["elasticnet__l1_ratio"])
    return gs.best_estimator_.predict(te[ENET_FEATS])

kf, gkf = KFold(5, shuffle=True, random_state=SEED), GroupKFold(5)
for split, by_gene in [("random5", False), ("genegroup5", True)]:
    for drug in DRUGS:
        d = TAB[TAB.treatment == drug]
        for tr_idx, te_idx in (gkf.split(d, groups=d.GENEINFO) if by_gene else kf.split(d)):
            tr, te = d.iloc[tr_idx], d.iloc[te_idx]
            rho, rmse = score(te, enet_tuned(tr, te, f"{drug}"))
            rows.append(dict(split=split, model="enet_tuned", drug=drug,
                             n_test=len(te), spearman=rho, rmse_log=rmse))

for drug in DRUGS:  # cftr_holdout (enet)
    tr = TAB[(TAB.treatment == drug) & (TAB.GENEINFO != "CFTR")]
    te = TAB[(TAB.treatment == drug) & (TAB.GENEINFO == "CFTR")]
    rho, rmse = score(te, enet_tuned(tr, te, drug))
    rows.append(dict(split="cftr_holdout", model="enet_tuned", drug=drug,
                     n_test=len(te), spearman=rho, rmse_log=rmse))

# ---------- 2а) мультитаск HGB, обычные сплиты (фолды по вариантам) ----------
var_meta = TAB.drop_duplicates("mutation_identifier").set_index("mutation_identifier").GENEINFO
for split in ["random5", "genegroup5"]:
    for drug in DRUGS:
        d = TAB[TAB.treatment == drug]
        vids = d.mutation_identifier.unique()
        if split == "random5":
            splits = KFold(5, shuffle=True, random_state=SEED).split(vids)
        else:
            splits = GroupKFold(5).split(vids, groups=pd.Series(vids).map(var_meta))
        for tr_v, te_v in splits:
            trv, tev = set(vids[tr_v]), set(vids[te_v])
            tr = TAB[TAB.treatment.isin(DRUGS) & TAB.mutation_identifier.isin(trv)]
            te = d[d.mutation_identifier.isin(tev)]
            rho, rmse = score(te, hgb_fit_predict(tr, te, tr.treatment, te.treatment))
            rows.append(dict(split=split, model="mt_hgb", drug=drug,
                             n_test=len(te), spearman=rho, rmse_log=rmse))

for drug in DRUGS:  # cftr_holdout (мультитаск: train = все препараты не-CFTR вариантов)
    tr = TAB[TAB.treatment.isin(DRUGS) & (TAB.GENEINFO != "CFTR")]
    te = TAB[(TAB.treatment == drug) & (TAB.GENEINFO == "CFTR")]
    rho, rmse = score(te, hgb_fit_predict(tr, te, tr.treatment, te.treatment))
    rows.append(dict(split="cftr_holdout", model="mt_hgb", drug=drug,
                     n_test=len(te), spearman=rho, rmse_log=rmse))

# ---------- 2б) few-shot LODO: held-out препарат через профиль корреляций ----------
piv = TAB.pivot_table(index="mutation_identifier", columns="treatment", values="y")
for held in DRUGS:
    others = [c for c in DRUGS if c != held]
    cand = piv.dropna(subset=[held])
    calib_ids = cand.sample(min(300, len(cand)), random_state=SEED).index
    C = piv.loc[calib_ids].corr(method="spearman")      # корреляции на калибровке (few-shot бюджет)
    tr = TAB[TAB.treatment.isin(others) & ~TAB.mutation_identifier.isin(
        TAB[(TAB.treatment == held) & ~TAB.mutation_identifier.isin(calib_ids)].mutation_identifier)]  # фикс утечки (аудит 06, п.1): тест-варианты не должны быть в train под другими препаратами
    te = TAB[(TAB.treatment == held) & ~TAB.mutation_identifier.isin(calib_ids)]
    Xtr, Xte = prep(tr), prep(te)
    for h in others:                                     # 7 фич: corr(препарат строки, h)
        Xtr[f"prof_{h}"] = tr.treatment.map(C[h]).values
        Xte[f"prof_{h}"] = C.loc[held, h]
    for c in CATS:
        Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                      categorical_features="from_dtype", random_state=SEED)
    m.fit(Xtr, tr.y)
    rho, rmse = score(te, m.predict(Xte))
    rows.append(dict(split="lodo_fewshot", model="mt_hgb_profile", drug=held,
                     n_test=len(te), spearman=rho, rmse_log=rmse))

res = pd.DataFrame(rows)
res.to_csv("results/metrics_stage2.csv", index=False)
summ = (res.groupby(["split", "model"])
          .agg(spearman_mean=("spearman", "mean"), spearman_min=("spearman", "min"),
               rmse_log_mean=("rmse_log", "mean"), n=("spearman", "size"))
          .round(3).reset_index())
summ.to_csv("results/summary_stage2.csv", index=False)
with open("results/summary_stage2.md", "w") as f:
    f.write("# Этап 2: тюнинг enet + drug-aware мультитаск + few-shot LODO\n\n")
    f.write(summ.to_string(index=False))
    f.write("\n\nКонтекст: бустинг per-drug (этап 1) = 0.651/0.649/0.630 на random5/genegroup5/cftr_holdout; "
            "lodo ridge floor (этап 1) = 0.751, но он требует измерений 7 препаратов того же варианта. "
            "mt_hgb_profile использует только последовательность + 300 калибровочных измерений held-out препарата.\n")
print(summ.to_string(index=False))
print("\nпример лучших гиперпараметров enet:",
      dict(list(best_hp.items())[:4]))

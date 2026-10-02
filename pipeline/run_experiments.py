# Шаг 2: бейзлайны + LOD-CV фреймворк.
# Модели: rules (медианы), elastic net (reimpl. статьи), градиентный бустинг.
# Сплиты: random5 | genegroup5 (главный) | cftr_holdout | lodo (zero-shot по препарату).
# Выход: results/metrics_baselines.csv + results/summary_baselines.md
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

SEED = 0
TAB = pd.read_parquet("data/model_table.parquet")
TAB = TAB.dropna(subset=["RT"]).copy()
TAB["GENEINFO"] = TAB.GENEINFO.astype(str).replace("None", "NO_GENE")  # вирусные контроли без гена
TAB["y"] = np.log10(TAB["RT"])
DRUGS = sorted([t for t in TAB.treatment.unique() if t != "Untreated"])
GENES = TAB.GENEINFO

CATS = ["stop_type", "up_123nt", "down_123nt", "up_codon_2", "up_aa"]
NUMS = ["aa_gene_position", "nt_CDS_pos", "GC%", "deltaG_OA", "deltaG_mFold",
        "tAI_up_1aa", "CAI_up_1aa", "dist_to_downstreamstop"]
ENET_FEATS = ["stop_type", "up_123nt", "down_123nt"]          # как в статье: ±3 нт + тип стопа

def fit_predict(model_name, drug, tr, te):
    """Обучить модель на tr (строки одного препарата), предсказать te. Возврат y_pred."""
    if model_name == "rules":
        med = tr.groupby(["stop_type", "down_1nt"])["y"].median()
        glob = tr["y"].median()
        key = list(zip(te.stop_type, te.down_1nt))
        return np.array([med.get(k, glob) for k in key])
    Xtr, Xte = tr[ENET_FEATS], te[ENET_FEATS]
    if model_name == "enet":
        pipe = make_pipeline(
            ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), ENET_FEATS)]),
            ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=SEED))
    elif model_name == "hgb":
        def prep(s):
            X = s[CATS + NUMS].copy()
            for c in NUMS:  # строковые 'NA'/None -> NaN
                X[c] = pd.to_numeric(X[c], errors="coerce")
            for c in CATS:
                X[c] = X[c].astype(str).fillna("NA").astype("category")
            return X
        Xtr, Xte = prep(tr), prep(te)
        for c in CATS:  # согласовать категории train/test
            Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                          categorical_features="from_dtype",
                                          random_state=SEED)
        m.fit(Xtr, tr.y)
        return m.predict(Xte)
    pipe.fit(Xtr, tr.y)
    return pipe.predict(Xte)

def score(te, pred):
    mask = ~np.isnan(pred)
    rho = spearmanr(te.y[mask], pred[mask]).statistic
    rmse = float(np.sqrt(np.mean((te.y[mask] - pred[mask]) ** 2)))
    return rho, rmse

rows = []
def evaluate(split, models, folds):
    """folds: итератор (train_idx, test_idx) по строкам одного препарата."""
    for drug in DRUGS + ["Untreated"]:
        d = TAB[TAB.treatment == drug]
        for tr_idx, te_idx in folds(d):
            tr, te = d.iloc[tr_idx], d.iloc[te_idx]
            for m in models:
                rho, rmse = score(te, fit_predict(m, drug, tr, te))
                rows.append(dict(split=split, model=m, drug=drug, n_test=len(te),
                                 spearman=rho, rmse_log=rmse))

# --- сплит 1: случайный 5-fold (сравнимость со статьёй) ---
kf = KFold(5, shuffle=True, random_state=SEED)
evaluate("random5", ["rules", "enet", "hgb"], lambda d: kf.split(d))

# --- сплит 2: leave-genes-out, GroupKFold по гену (главный) ---
gkf = GroupKFold(5)
evaluate("genegroup5", ["rules", "enet", "hgb"],
         lambda d: gkf.split(d, groups=d.GENEINFO))

# --- сплит 3: CFTR holdout: train = все не-CFTR, test = 76 CFTR-вариантов ---
cftr_rows = []
wide_pred, wide_true = {}, {}
for drug in DRUGS:
    tr = TAB[(TAB.treatment == drug) & (TAB.GENEINFO != "CFTR")]
    te = TAB[(TAB.treatment == drug) & (TAB.GENEINFO == "CFTR")]
    for m in ["rules", "enet", "hgb"]:
        rho, rmse = score(te, fit_predict(m, drug, tr, te))
        rows.append(dict(split="cftr_holdout", model=m, drug=drug, n_test=len(te),
                         spearman=rho, rmse_log=rmse))
        cftr_rows.append(dict(drug=drug, model=m, spearman=rho, rmse_log=rmse))
    wide_pred[drug] = dict(zip(te.mutation_identifier_short, fit_predict("hgb", drug, tr, te)))
    wide_true[drug] = dict(zip(te.mutation_identifier_short, te.y))

# top-1 точность выбора лучшего препарата (HGB, CFTR holdout)
ids = sorted(set.intersection(*(set(wide_pred[d]) for d in DRUGS)))
if ids:
    p = np.array([[wide_pred[d][i] for d in DRUGS] for i in ids])
    t = np.array([[wide_true[d][i] for d in DRUGS] for i in ids])
    top1 = float((p.argmax(1) == t.argmax(1)).mean())
    top2 = float(np.mean([t.argmax(1)[k] in np.argsort(-p[k])[:2] for k in range(len(ids))]))
    rows.append(dict(split="cftr_holdout_topdrug", model="hgb", drug="top1",
                     n_test=len(ids), spearman=top1, rmse_log=top2))

# --- сплит 4: LODO — zero-shot на невидимый препарат ---
# floor-бейзлайн без последовательности: ridge-регрессия RT_holdout ~ RT остальных 7 препаратов
piv = TAB.pivot_table(index="mutation_identifier", columns="treatment", values="y")
half = piv.dropna(thresh=5)
tr_v, te_v = half.iloc[: len(half) // 2], half.iloc[len(half) // 2:]
medians = tr_v.median()                       # импутация NaN медианами train-половины
tr_v, te_v = tr_v.fillna(medians), te_v.fillna(medians)
for held in DRUGS:
    others = [c for c in DRUGS if c != held]
    r = Ridge(alpha=1.0).fit(tr_v[others], tr_v[held])
    pred = r.predict(te_v[others])
    rho, rmse = score(te_v.reset_index(drop=True).assign(y=te_v[held].values), pred)
    rows.append(dict(split="lodo_ridge", model="ridge_7drugs", drug=held,
                     n_test=int(te_v[held].notna().sum()), spearman=rho, rmse_log=rmse))

res = pd.DataFrame(rows)
res.to_csv("results/metrics_baselines.csv", index=False)

# --- сводка ---
summ = (res.groupby(["split", "model"])
          .agg(spearman_mean=("spearman", "mean"),
               spearman_min=("spearman", "min"),
               rmse_log_mean=("rmse_log", "mean"), n=("spearman", "size"))
          .round(3).reset_index())
summ.to_csv("results/summary_baselines.csv", index=False)
with open("results/summary_baselines.md", "w") as f:
    f.write("# Бейзлайны и LOD-CV (промежуточный итог)\n\n")
    f.write(summ.to_string(index=False))
    f.write("\n\nЗаметки: random5 — сравнимость со статьёй (случайный сплит, вариант одного гена "
            "может быть и в train, и в test); genegroup5 — перенос на невидимые гены; "
            "cftr_holdout — train без CFTR, тест на 76 измеренных CFTR; "
            "lodo_ridge — zero-shot floor для невидимого препарата (без признаков последовательности, "
            "только межпрепаратная корреляция; настоящий drug-aware тест — этап 2, H2).\n")
print(summ.to_string(index=False))
print("\nresults/metrics_baselines.csv записан; строк:", len(res))

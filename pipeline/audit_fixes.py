# Верификация аудита (чекпоинт 06): пересчёт всех заявленных в нём цифр.
# Выход: results/audit_verification.csv
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyreadr
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

out = []
def add(name, value, note=""):
    out.append(dict(check=name, value=round(float(value), 3), note=note))
    print(f"{name:55s} {value:.3f}  {note}", flush=True)

# ---------- A. шумовой потолок: согласование репликат 1 vs 2 ----------
src = pyreadr.read_r("toledano_data/treated_samples.rds")[None]
for c in ["treatment", "GENEINFO"]:
    if str(src[c].dtype) == "category":
        src[c] = src[c].astype(str)
piv = src.pivot_table(index="mutation_identifier", columns=["treatment", "replicate"], values="RT")
rhos = []
for t in piv.columns.get_level_values(0).unique():
    a, b = piv[(t, 1)], piv[(t, 2)]
    ok = a.notna() & b.notna()
    if ok.sum() > 50:
        r = spearmanr(a[ok], b[ok]).statistic
        rhos.append(r)
        if t in ("Untreated", "G418", "DAP"):
            add(f"A rep1_vs_rep2 [{t}]", r, f"n={ok.sum()}")
drugs8 = [t for t in piv.columns.get_level_values(0).unique() if t != "Untreated"]
add("A rep_ceiling_mean_8drugs", np.mean([spearmanr(piv[(t,1)], piv[(t,2)]).statistic
        for t in drugs8]), "потолок для random5/genegroup5")

# ---------- B. средние по 8 препаратам (этап 1) ----------
m = pd.read_csv("results/metrics_baselines.csv")
m8 = m[m.drug != "Untreated"]
for (split, model), g in m8.groupby(["split", "model"]):
    if split in ("random5", "genegroup5", "cftr_holdout") and model == "hgb":
        add(f"B stage1_8drugs [{split}, hgb]", g.spearman.mean(),
            "было с Untreated: " + str(round(m[(m.split==split)&(m.model=='hgb')].spearman.mean(),3)))

# ---------- C. мажоритарные бейзлайны top-1/top-2 (CFTR) ----------
cf = pd.read_csv("data/cftr_measured.csv", index_col=0)
drugs = [c for c in cf.columns if c != "Untreated"]
best = cf[drugs].idxmax(axis=1).value_counts()
maj1 = best.max() / best.sum()
freq = (best / best.sum()).sort_values(ascending=False)
maj2 = cf[drugs].idxmax(axis=1).isin(freq.index[:2]).mean()
add("C majority_top1 (always DAP)", maj1, f"частоты: {dict(freq.round(3))}")
add("C majority_top2 (DAP|SJ6986)", maj2, "против HGB 0.592/0.803")

# ---------- D. CFTR-only LOO (one-hot ±3 нт, ridge) ----------
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
TAB = pd.read_parquet("data/model_table.parquet").dropna(subset=["RT"]).copy()
TAB["y"] = np.log10(TAB.RT)
cftr = TAB[TAB.GENEINFO.astype(str) == "CFTR"]
rhos_d = []
for drug in drugs:
    d = cftr[cftr.treatment == drug].dropna(subset=["y"])
    X = d[["stop_type", "up_123nt", "down_123nt"]].astype(str)
    pipe = make_pipeline(ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"),
                                             ["stop_type", "up_123nt", "down_123nt"])]),
                         Ridge(alpha=1.0))
    preds = np.empty(len(d))
    for i in range(len(d)):  # LOO
        mask = np.ones(len(d), bool); mask[i] = False
        pipe.fit(X[mask], d.y.values[mask])
        preds[i] = pipe.predict(X.iloc[[i]])[0]
    ok = ~np.isnan(preds)
    rhos_d.append(spearmanr(d.y.values[ok], preds[ok]).statistic)
add("D CFTRonly_LOO_mean_8drugs", np.mean(rhos_d),
    f"per-drug min {min(rhos_d):.2f}; против геномной модели 0.614-0.635")

# ---------- E. чиненый LODO: тест-варианты вне train + потолок ----------
CATS = ["stop_type", "up_123nt", "down_123nt", "up_codon_2", "up_aa"]
NUMS = ["aa_gene_position", "nt_CDS_pos", "GC%", "deltaG_OA", "deltaG_mFold",
        "tAI_up_1aa", "CAI_up_1aa", "dist_to_downstreamstop"]
from sklearn.ensemble import HistGradientBoostingRegressor
def prep(s):
    X = s[CATS + NUMS].copy()
    for c in NUMS: X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in CATS: X[c] = X[c].astype(str).fillna("NA").astype("category")
    return X
pivv = TAB.pivot_table(index="mutation_identifier", columns="treatment", values="y")
fixed, pooled, ceil = [], [], []
for held in drugs:
    others = [c for c in drugs if c != held]
    cand = pivv.dropna(subset=[held])
    calib_ids = cand.sample(min(300, len(cand)), random_state=0).index
    C = pivv.loc[calib_ids].corr(method="spearman")
    te = TAB[(TAB.treatment == held) & ~TAB.mutation_identifier.isin(calib_ids)]
    tr = TAB[TAB.treatment.isin(others) & ~TAB.mutation_identifier.isin(te.mutation_identifier)]  # фикс утечки
    Xtr, Xte = prep(tr), prep(te)
    for h in others:
        Xtr[f"prof_{h}"] = tr.treatment.map(C[h]).values
        Xte[f"prof_{h}"] = C.loc[held, h]
    for c in CATS:
        Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)
    mm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                       categorical_features="from_dtype", random_state=0)
    mm.fit(Xtr, tr.y)
    p = mm.predict(Xte)
    fixed.append(spearmanr(te.y, p).statistic)
    # потолок: среднее RT варианта по 7 другим препаратам (измерено)
    vm = pivv.loc[te.mutation_identifier, others].mean(axis=1)
    ok = vm.notna().values
    ceil.append(spearmanr(te.y.values[ok], vm.values[ok]).statistic)
    # без профиля
    Xtr2, Xte2 = prep(tr), prep(te)
    for c in CATS:
        Xte2[c] = pd.Categorical(Xte2[c], categories=Xtr2[c].cat.categories)
    m2 = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                       categorical_features="from_dtype", random_state=0)
    m2.fit(Xtr2, tr.y)
    pooled.append(spearmanr(te.y, m2.predict(Xte2)).statistic)
add("E lodo_fixed_profile", np.mean(fixed), "было 0.654 (утечка)")
add("E lodo_fixed_pooled_nodrug", np.mean(pooled), "вклад профиля = E - это")
add("E ceiling_variant_mean_7drugs", np.mean(ceil), "потолок информационного режима (аудит: 0.628)")

# ---------- F. TransStop model-free бейзлайн на их тест-сплите ----------
DRUG_ORDER = ["FUr", "Gentamicin", "CC90009", "G418", "Clitocine", "DAP", "SJ6986", "SRI", "Untreated"]
fr = []
for name in DRUG_ORDER:
    d = src[(src.treatment == name) & (src.replicate == 2) & (src.reads_allbins > 15)
            & (src.viral.astype(str) == "no") & src.RT.notna()]
    fr.append(d[["mutation_identifier", "RT", "GENEINFO"]].assign(treatment=name))
master = pd.concat(fr, ignore_index=True)
tr_rows, te_rows = train_test_split(master, test_size=0.1, random_state=42, stratify=master.treatment)
W = master.pivot_table(index="mutation_identifier", columns="treatment", values="RT")
teW = te_rows.pivot_table(index="mutation_identifier", columns="treatment", values="RT")
Wtr = tr_rows.pivot_table(index="mutation_identifier", columns="treatment", values="RT")
preds, labels = [], []
for held in DRUG_ORDER:
    others = [c for c in DRUG_ORDER if c != held]
    tr_ids = Wtr.dropna(subset=[held]).index
    X = Wtr.loc[tr_ids, others].copy(); y = Wtr.loc[tr_ids, held]
    X = X.fillna(X.mean())
    r = Ridge(alpha=1.0).fit(X, y)
    te_ids = teW[teW[held].notna()].index
    Xt = W.loc[te_ids, others].fillna(X.mean())
    preds += r.predict(Xt).tolist()
    labels += W.loc[te_ids, held].tolist()
preds = np.array(preds); labels = np.array(labels)
preds[preds < 0] = 0
add("F transstop_modelfree_R2", r2_score(labels, preds),
    "перенос RT варианта с других препаратов; аудит: 0.648; TransStop: 0.940")

pd.DataFrame(out).to_csv("results/audit_verification.csv", index=False)
print("\n-> results/audit_verification.csv")

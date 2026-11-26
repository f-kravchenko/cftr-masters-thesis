# Каталог: in silico-оценка readthrough для панели нонсенс-вариантов CFTR из ClinVar (312 записей).
# Признаки — как у enet_tuned (тип стоп-кодона, ±3 нт); строятся из записи c. в названии варианта
# и транскрипта NM_000492.4. Модель обучается без CFTR (условия cftr_holdout), поэтому заявленная
# точность (Spearman на 76 измеренных вариантах) относится к той же модели, что выдаёт каталог.
# Самопроверка: признаки 76 измеренных CFTR-вариантов, заново собранные из транскрипта и аллелей,
# совпадают с таблицей Toledano.
# Выход: results/catalog_cftr_readthrough.csv + results/catalog_cftr_readthrough.md
import warnings; warnings.filterwarnings("ignore")
import re
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

SEED = 0
ENET_FEATS = ["stop_type", "up_123nt", "down_123nt"]
STOPS = {"TAA", "TAG", "TGA"}
AA3 = dict(zip("Ala Arg Asn Asp Cys Gln Glu Gly His Ile Leu Lys Met Phe Pro Ser Thr Trp Tyr Val".split(), "ARNDCQEGHILKMFPSTWYV"))
IUPAC = {"R": "AG"}

# ---------- 1) транскрипт NM_000492.4 (GenBank): CDS и контекст стоп-кодона ----------
gb = open("data/cftr_NM_000492.4.gb").read()
cds_start, cds_end = map(int, re.search(r"^     CDS             (\d+)\.\.(\d+)", gb, re.M).groups())
mrna = "".join(re.findall(r"[acgtn]+", gb.split("ORIGIN", 1)[1].split("//")[0])).upper()
cds = mrna[cds_start - 1:cds_end]
assert cds[:3] == "ATG" and cds[-3:] in STOPS and len(cds) % 3 == 0, "CDS looks wrong"
assert not any(cds[i:i + 3] in STOPS for i in range(0, len(cds) - 3, 3)), "internal stop codon in CDS"

def rna(s):                                        # как в таблице Toledano: строчные, с U
    return s.lower().replace("t", "u")

def context(pos):                                   # pos = номер кодона (1-based), в котором возник стоп
    a = cds_start - 1 + 3 * (pos - 1)               # координата кодона в мРНК (5'UTR доступна для pos = 1)
    return rna(mrna[a - 3:a]), rna(mrna[a:a + 3]), rna(mrna[a + 3:a + 6])

# ---------- 2) самопроверка на 76 измеренных CFTR-вариантах ----------
TAB = pd.read_parquet("data/model_table.parquet").dropna(subset=["RT"]).copy()
TAB["GENEINFO"] = TAB.GENEINFO.astype(str).replace("None", "NO_GENE")
DRUGS = sorted(t for t in TAB.treatment.unique() if t != "Untreated")
cftr = TAB[TAB.GENEINFO == "CFTR"].drop_duplicates("mutation_identifier_short")
ok = 0
for _, r in cftr.iterrows():
    pos = int(r.aa_gene_position)
    ref_codon = cds[3 * (pos - 1):3 * pos]
    up, _, down = context(pos)
    # тип стопа восстанавливается из референсного кодона и аллелей (CFTR на +‑цепи: аллели совпадают с мРНК)
    made = [ref_codon[:i] + r.Mutant_allele + ref_codon[i + 1:] for i in range(3) if ref_codon[i] == r.Ref_allele]
    ok += (up == r.up_123nt) and (down == r.down_123nt) and (sum(rna(m) == r.stop_type for m in made) == 1)
assert ok == len(cftr) == 76, f"self-check failed: {ok} of {len(cftr)} measured CFTR variants reproduced"
print(f"самопроверка: контекст и тип стопа восстановлены для {ok} из {len(cftr)} измеренных CFTR-вариантов", flush=True)

# ---------- 3) модель: elastic net по каждому препарату, обучение без CFTR ----------
def enet_tuned(tr):                                # как в run_stage2.py
    gs = GridSearchCV(make_pipeline(ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), ENET_FEATS)]),
                                    ElasticNet(max_iter=20000, random_state=SEED)),
                      {"elasticnet__alpha": np.logspace(-4, 1, 12), "elasticnet__l1_ratio": [0.1, 0.5, 0.9]},
                      cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1).fit(tr[ENET_FEATS], np.log10(tr.RT))
    return gs.best_estimator_

models, rho, rmse = {}, {}, {}
for d in DRUGS:
    tr = TAB[(TAB.treatment == d) & (TAB.GENEINFO != "CFTR")]
    te = TAB[(TAB.treatment == d) & (TAB.GENEINFO == "CFTR")]
    models[d] = enet_tuned(tr)
    p = models[d].predict(te[ENET_FEATS])
    rho[d], rmse[d] = spearmanr(np.log10(te.RT), p).statistic, float(np.sqrt(np.mean((np.log10(te.RT) - p) ** 2)))
train_vals = {f: set(TAB[TAB.GENEINFO != "CFTR"][f]) for f in ENET_FEATS}
print("cftr_holdout: Spearman %.3f, RMSE(log10) %.3f" % (np.mean(list(rho.values())), np.mean(list(rmse.values()))), flush=True)

# ---------- 4) панель ClinVar -> признаки -> предсказания ----------
cv = pd.read_csv("data/clinvar_cftr_nonsense.tsv", sep="\t")
rows = []
for _, r in cv.iterrows():
    pos, ref, alt = re.search(r"c\.(\d+)([ACGT])>([ACGT]|[RYKMSWBDHVN])", r.Name).groups()
    paa, ppos = re.search(r"p\.([A-Z][a-z]{2})(\d+)Ter", r.Name).groups()
    for a in IUPAC.get(alt, alt):                   # c.3276C>R: обе возможные замены
        i, codon_start = int(pos) - 1, 3 * (int(ppos) - 1)
        assert cds[i] == ref and codon_start <= i < codon_start + 3, r.Name
        mut = cds[codon_start:i] + a + cds[i + 1:codon_start + 3]
        assert mut in STOPS, r.Name
        up, _, down = context(int(ppos))
        rows.append(dict(allele_id=r.AlleleID, name=r.Name, protein=f"{AA3[paa]}{ppos}X",
                         cds_change=f"c.{pos}{ref}>{a}", aa_pos=int(ppos),
                         clinical_significance=r.ClinicalSignificance, review_status=r.ReviewStatus,
                         stop_type=rna(mut), up_123nt=up, down_123nt=down, nt_ambiguous=alt not in "ACGT"))
panel = pd.DataFrame(rows)
for d in DRUGS:
    panel[f"RT_{d}"] = 10 ** models[d].predict(panel[ENET_FEATS])

# ---------- 5) каталог: лучший препарат, интервал, измерения, новизна контекста ----------
meas = pd.read_csv("data/cftr_measured.csv", index_col=0)
panel["measured"] = panel.protein.isin(meas.index)
panel["RT_measured_best"] = meas[DRUGS].max(axis=1).reindex(panel.protein).round(2).values
panel["measured_best_drug"] = meas[DRUGS].idxmax(axis=1).reindex(panel.protein).values
R = panel[[f"RT_{d}" for d in DRUGS]].values
order = np.argsort(-R, axis=1)
panel["best_drug"] = np.array(DRUGS)[order[:, 0]]
panel["RT_best"] = R.max(1)
panel["second_drug"] = np.array(DRUGS)[order[:, 1]]
panel["RT_second"] = R[np.arange(len(R)), order[:, 1]]
panel["RT_best_lo"] = [max(0.0, 10 ** (np.log10(b) - 1.96 * rmse[d])) for b, d in zip(panel.RT_best, panel.best_drug)]
panel["RT_best_hi"] = [10 ** (np.log10(b) + 1.96 * rmse[d]) for b, d in zip(panel.RT_best, panel.best_drug)]
panel["unseen_fields"] = [sum(panel.at[i, f] not in train_vals[f] for f in ENET_FEATS) for i in panel.index]
panel["context"] = [(u + s + d).upper().replace("U", "T") for u, s, d in zip(panel.up_123nt, panel.stop_type, panel.down_123nt)]
num = [c for c in panel if c.startswith("RT_")]
panel[num] = panel[num].round(2)
panel = panel.sort_values(["RT_best", "aa_pos"], ascending=[False, True]).reset_index(drop=True)
panel.insert(0, "rank", np.arange(1, len(panel) + 1))

cols = ["rank", "allele_id", "protein", "cds_change", "aa_pos", "context", "clinical_significance", "review_status",
        "best_drug", "RT_best", "RT_best_lo", "RT_best_hi", "second_drug", "RT_second", "measured",
        "RT_measured_best", "measured_best_drug", "unseen_fields", "nt_ambiguous"] + [f"RT_{d}" for d in DRUGS]
panel[cols].to_csv("results/catalog_cftr_readthrough.csv", index=False)

# ---------- 6) сводка ----------
m, new = panel[panel.measured], panel[~panel.measured]
top1 = (m.best_drug == m.measured_best_drug).mean()
top2 = m.apply(lambda r: r.measured_best_drug in [r.best_drug, r.second_drug], axis=1).mean()
maj1 = (m.measured_best_drug == "DAP").mean()      # мажоритарные бейзлайны на тех же строках
maj2 = m.measured_best_drug.isin(["DAP", "SJ6986"]).mean()
show = new.head(10)[["protein", "context", "best_drug", "RT_best", "RT_best_lo", "RT_best_hi", "second_drug", "unseen_fields"]]
with open("results/catalog_cftr_readthrough.md", "w") as f:
    f.write("# Каталог readthrough для нонсенс-вариантов CFTR (панель ClinVar)\n\n")
    f.write("Модель: elastic net на типе стоп-кодона и ±3 нт (как enet_tuned), отдельная по каждому из 8 препаратов, "
            "обучена на данных Toledano 2024 без CFTR-вариантов. Признаки построены из записей c. в названиях "
            "вариантов и транскрипта NM_000492.4 (RefSeq, MANE Select). Предсказывается репортёрный readthrough (%). "
            "`unseen_fields` — сколько из трёх признаков варианта не встречались в обучении (0–3).\n\n")
    f.write(f"**Самопроверка признаков:** контекст и тип стопа, восстановленные из транскрипта и аллелей, совпали "
            f"с таблицей Toledano для {ok} из {len(cftr)} измеренных CFTR-вариантов.\n\n")
    f.write(f"**Точность той же модели на 76 измеренных вариантах (они не входили в обучение):** "
            f"Spearman {np.mean(list(rho.values())):.3f}; RMSE {np.mean(list(rmse.values())):.3f} log10; "
            f"лучший препарат совпал с измеренным в {top1:.0%} случаев, вошёл в топ-2 — в {top2:.0%} "
            f"(на тех же строках правило «всегда DAP» даёт {maj1:.0%}, пара «DAP или SJ6986» — {maj2:.0%}). "
            f"Измерения Toledano сопоставлены по белковой замене.\n\n")
    f.write(f"**Состав:** {len(panel)} строк каталога: {int(panel.measured.sum())} с измерениями ({m.protein.nunique()} вариантов) "
            f"и {len(new)} новых ({new.protein.nunique()} вариантов); у {int((new.unseen_fields > 0).sum())} новых строк "
            f"какой-то из признаков не встречался в обучении.\n\n")
    f.write("## Топ-10 новых вариантов по предсказанному лучшему readthrough\n\n")
    f.write(show.to_string(index=False) + "\n\n")
    f.write("## Ограничения\n\n"
            "- Предсказывается readthrough в репортёрной системе (HEK293T, sort-seq); это не клинический ответ: "
            "на органоидах (Destoop 2026, 7 вариантов) репортёрный readthrough не предсказал ответ (−0.22).\n"
            "- Только 8 препаратов из набора данных; ataluren и ELX-02 в данных отсутствуют.\n"
            f"- Выбор препарата линейная модель почти не улучшает: лучший препарат совпадает с измеренным в {top1:.0%} "
            f"против {maj1:.0%} у правила «всегда DAP» (у бустинга с расширенными признаками было 59% против 43%). "
            f"Каталог ранжирует варианты; из препаратов он скорее сужает выбор до двух ({top2:.0%} против {maj2:.0%}).\n"
            "- Точность ограничена: Spearman ≈ 0.61, типичная ошибка — примерно вдвое. Это ранжирование для "
            "экспериментальной проверки, а не прогноз для пациента.\n"
            f"- {new.protein.nunique()} новых вариантов не имеют измерений; интервал — это ±1.96·RMSE модели на отложенном CFTR.\n")
print(f"каталог: {len(panel)} строк -> results/catalog_cftr_readthrough.csv", flush=True)
print(show.head(5).to_string(index=False), flush=True)

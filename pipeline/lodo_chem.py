# Частная H2 (zero-shot препарат): LODO, где препарат представлен ТОЛЬКО молекулярными
# дескрипторами RDKit из SMILES (PubChem). Ни identity, ни измерений held-out препарата.
# Порог H2: mean Spearman >= 0.42 (= 0.8 x профильного few-shot 0.518).
# Выход: results/metrics_lodo_chem.csv + results/summary_lodo_chem.md
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# SMILES (PubChem, ConnectivitySMILES; data/pubchem_smiles.tsv)
SMILES = {
    "Gentamicin": "CC(C1CCC(C(O1)OC2C(CC(C(C2O)OC3C(C(C(CO3)(C)O)NC)O)N)N)N)NC",
    "G418": "CC(C1C(C(C(C(O1)OC2C(CC(C(C2O)OC3C(C(C(CO3)(C)O)NC)O)N)N)N)O)O)O",
    "Clitocine": "C1=NC(=C(C(=N1)NC2C(C(C(O2)CO)O)O)[N+](=O)[O-])N",
    "FUr": "C1=C(C(=O)NC(=O)N1C2C(C(C(O2)CO)O)O)F",
    "DAP": "C1=NC2=NC(=NC(=C2N1)N)N",
    "SRI": "CN1C2=CC=CC=C2C(=O)C3=C1N=C(N(C3=O)C4=CC=CC=C4)C5CCC5",
    "SJ6986": "C1CC(=O)NC(=O)C1N2C(=O)C3=C(C2=O)C=C(C=C3)NS(=O)(=O)C4=CC=CC=C4OC(F)(F)F",
    "CC90009": "C1CC(=O)NC(=O)C1N2CC3=C(C2=O)C=CC(=C3)CNC(=O)C(C4=CC=C(C=C4)Cl)(F)F",
    "Ataluren": "C1=CC=C(C(=C1)C2=NC(=NO2)C3=CC(=CC=C3)C(=O)O)F",
    "ELX02": "CC(C1C(C(C(O1)OC2C(C(CC(C2OC3C(C(C(C(O3)C(C)O)O)O)N)N)N)O)O)O)N",
}

def descr(smiles):
    m = Chem.MolFromSmiles(smiles)
    ha = m.GetNumHeavyAtoms()
    return dict(
        MolWt=Descriptors.MolWt(m), TPSA=rdMolDescriptors.CalcTPSA(m),
        MolLogP=Descriptors.MolLogP(m), HBD=rdMolDescriptors.CalcNumHBD(m),
        HBA=rdMolDescriptors.CalcNumHBA(m), RotBonds=rdMolDescriptors.CalcNumRotatableBonds(m),
        Rings=rdMolDescriptors.CalcNumRings(m), AromaticProp=rdMolDescriptors.CalcNumAromaticRings(m) / max(1, rdMolDescriptors.CalcNumRings(m)),
        FracCSP3=rdMolDescriptors.CalcFractionCSP3(m), HeavyAtoms=ha, FormalCharge=Chem.GetFormalCharge(m))

DRX = pd.DataFrame({d: descr(s) for d, s in SMILES.items()}).T
DRX.to_csv("data/drug_descriptors.csv")
print("дескрипторы препаратов:\n", DRX.round(2).to_string(), flush=True)

TAB = pd.read_parquet("data/model_table.parquet").dropna(subset=["RT"]).copy()
TAB["GENEINFO"] = TAB.GENEINFO.astype(str).replace("None", "NO_GENE")
TAB["y"] = np.log10(TAB.RT)
DRUGS = sorted([t for t in TAB.treatment.unique() if t != "Untreated"])
DESCR_COLS = list(DRX.columns)
for c in DESCR_COLS:
    TAB[c] = TAB.treatment.map(DRX[c])          # Untreated -> NaN (из LODO-обучения исключён)

CATS = ["stop_type", "up_123nt", "down_123nt", "up_codon_2", "up_aa"]
NUMS = ["aa_gene_position", "nt_CDS_pos", "GC%", "deltaG_OA", "deltaG_mFold",
        "tAI_up_1aa", "CAI_up_1aa", "dist_to_downstreamstop"]

def prep(s):
    X = s[CATS + NUMS + DESCR_COLS].copy()
    for c in NUMS + DESCR_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in CATS:
        X[c] = X[c].astype(str).fillna("NA").astype("category")
    return X

rows = []
for held in DRUGS:
    others = [d for d in DRUGS if d != held]                 # LODO: held-out препарата в train НЕТ
    te = TAB[(TAB.treatment == held)]
    tr = TAB[TAB.treatment.isin(others)
             & ~TAB.mutation_identifier.isin(te.mutation_identifier)]  # и тест-вариантов в train нет (аудит 06, п.1)
    tr = tr[tr[DESCR_COLS[0]].notna()]                       # Untreated без молекулы -> NaN -> исключён
    Xtr, Xte = prep(tr), prep(te)
    for c in CATS:
        Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                      categorical_features="from_dtype", random_state=0)
    m.fit(Xtr, tr.y)
    p = m.predict(Xte)
    rho = spearmanr(te.y, p).statistic
    rows.append(dict(drug=held, spearman=rho,
                     rmse_log=float(np.sqrt(np.mean((te.y - p) ** 2)))))
    print(f"held-out {held:10s} Spearman {rho:+.3f}", flush=True)

r = pd.DataFrame(rows)
r.to_csv("results/metrics_lodo_chem.csv", index=False)
mean = r.spearman.mean()
verdict = "ПОДТВЕРЖДЕНА (>=0.42)" if mean >= 0.42 else ("СЕРАЯ ЗОНА (0.1-0.42)" if mean >= 0.1 else "ОПРОВЕРГНУТА (<0.1)")
with open("results/summary_lodo_chem.md", "w") as f:
    f.write("# H2: zero-shot LODO из молекулярных дескрипторов (RDKit/PubChem SMILES)\n\n")
    f.write(r.round(3).to_string(index=False))
    f.write(f"\n\n**Средний Spearman: {mean:.3f}** — порог H2 >= 0.42: {verdict}\n\n"
            f"Контекст: профильный few-shot 0.518 (нужна калибрация 300 измерений), "
            f"pooled без признака препарата 0.474, потолок режима 0.628. "
            f"Дескрипторы не требуют НИ ОДНОГО измерения нового препарата.\n")
print(f"\nИТОГ H2: mean Spearman {mean:.3f} — {verdict}")

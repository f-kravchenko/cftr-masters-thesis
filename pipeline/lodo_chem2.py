# Улучшение zero-shot представления препарата (лестница):
#   V1 = MOA-класс (4 кластера) one-hot
#   V2 = V1 + 11 RDKit-дескрипторов
#   V3 = V2 + Morgan ECFP4 (2048) -> PCA(5)
# Всё на честном LODO: held-out препарат И тест-варианты вне train.
# Выход: results/metrics_lodo_chem2.csv
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

SMILES = dict(
    Gentamicin="CC(C1CCC(C(O1)OC2C(CC(C(C2O)OC3C(C(C(CO3)(C)O)NC)O)N)N)N)NC",
    G418="CC(C1C(C(C(C(O1)OC2C(CC(C(C2O)OC3C(C(C(CO3)(C)O)NC)O)N)N)N)O)O)O",
    Clitocine="C1=NC(=C(C(=N1)NC2C(C(C(O2)CO)O)O)[N+](=O)[O-])N",
    FUr="C1=C(C(=O)NC(=O)N1C2C(C(C(O2)CO)O)O)F",
    DAP="C1=NC2=NC(=NC(=C2N1)N)N",
    SRI="CN1C2=CC=CC=C2C(=O)C3=C1N=C(N(C3=O)C4=CC=CC=C4)C5CCC5",
    SJ6986="C1CC(=O)NC(=O)C1N2C(=O)C3=C(C2=O)C=C(C=C3)NS(=O)(=O)C4=CC=CC=C4OC(F)(F)F",
    CC90009="C1CC(=O)NC(=O)C1N2CC3=C(C2=O)C=CC(=C3)CNC(=O)C(C4=CC=C(C=C4)Cl)(F)F",
)
MOA = dict(Gentamicin="aminoglycoside", G418="aminoglycoside", ELX02="aminoglycoside",
           Clitocine="mrna_nucleoside", FUr="mrna_nucleoside", DAP="trna_mod",
           SRI="degrader_eRF1", SJ6986="degrader_eRF3", CC90009="degrader_eRF3")

def descr(s):
    m = Chem.MolFromSmiles(s)
    return [Descriptors.MolWt(m), rdMolDescriptors.CalcTPSA(m), Descriptors.MolLogP(m),
            rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
            rdMolDescriptors.CalcFractionCSP3(m), m.GetNumHeavyAtoms(), Chem.GetFormalCharge(m)]

DRUGS = sorted(SMILES)
DESC = pd.DataFrame({d: descr(s) for d, s in SMILES.items()}, index=[f"d{i}" for i in range(10)]).T
fps = {d: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)
       for d, s in SMILES.items()}
FP = pd.DataFrame({d: list(fps[d]) for d in DRUGS}).T
FP_PCA = pd.DataFrame(PCA(5, random_state=0).fit_transform(FP), index=FP.index,
                      columns=[f"pc{i}" for i in range(5)])
MOA_DF = pd.get_dummies(pd.Series(MOA)).astype(float)   # index=препараты, columns=классы

VARIANTS = {
    "V1_moa": MOA_DF,
    "V2_moa+descr": pd.concat([MOA_DF, DESC], axis=1),
    "V3_moa+descr+fppca": pd.concat([MOA_DF, DESC, FP_PCA], axis=1),
}

TAB = pd.read_parquet("data/model_table.parquet").dropna(subset=["RT"]).copy()
TAB["GENEINFO"] = TAB.GENEINFO.astype(str).replace("None", "NO_GENE")
TAB["y"] = np.log10(TAB.RT)
CATS = ["stop_type", "up_123nt", "down_123nt", "up_codon_2", "up_aa"]
NUMS = ["aa_gene_position", "nt_CDS_pos", "GC%", "deltaG_OA", "deltaG_mFold",
        "tAI_up_1aa", "CAI_up_1aa", "dist_to_downstreamstop"]

rows = []
for vname, feat in VARIANTS.items():
    T = TAB.copy()
    for c in feat.columns:
        T[c] = T.treatment.map(feat[c])
    for held in DRUGS:
        others = [d for d in DRUGS if d != held]
        te = T[T.treatment == held]
        tr = T[T.treatment.isin(others)
               & ~T.mutation_identifier.isin(te.mutation_identifier)]
        Xtr = tr[CATS + NUMS + list(feat.columns)].copy()
        Xte = te[CATS + NUMS + list(feat.columns)].copy()
        for c in NUMS + list(feat.columns):
            Xtr[c] = pd.to_numeric(Xtr[c], errors="coerce")
            Xte[c] = pd.to_numeric(Xte[c], errors="coerce")
        for c in CATS:
            Xtr[c] = Xtr[c].astype(str).fillna("NA").astype("category")
            Xte[c] = pd.Categorical(Xte[c].astype(str), categories=Xtr[c].cat.categories)
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                          categorical_features="from_dtype", random_state=0)
        m.fit(Xtr, tr.y)
        rho = spearmanr(te.y, m.predict(Xte)).statistic
        rows.append(dict(variant=vname, drug=held, spearman=rho))
    print(f"{vname}: mean {pd.DataFrame(rows).query('variant==@vname').spearman.mean():.3f}", flush=True)

r = pd.DataFrame(rows)
r.to_csv("results/metrics_lodo_chem2.csv", index=False)
print("\n", r.groupby("variant").spearman.agg(["mean", "min", "max"]).round(3).to_string())
print("\nконтекст: дескрипторы без MOA 0.142; профиль few-shot 0.518; потолок 0.628")

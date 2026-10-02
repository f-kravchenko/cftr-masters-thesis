# Шаг 1: сборка модельной таблицы из Toledano treated_samples.rds
# + ClinVar-панель нонсенс-вариантов CFTR (для будущего каталога).
# Выход: data/model_table.parquet, data/cftr_measured.csv, data/clinvar_cftr_nonsense.tsv
import warnings; warnings.filterwarnings("ignore")
import pyreadr, pandas as pd, urllib.request, gzip, os

TOLEDANO = "toledano_data/treated_samples.rds"
CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"

# колонки: таргет + признаки, статичные на уровне варианта (или пары вариант×препарат)
KEEP = ["mutation_identifier", "mutation_identifier_short", "GENEINFO", "treatment",
        "RT", "sd_RT",                      # таргет и его шум
        "stop_type", "up_1nt", "up_12nt", "up_123nt", "up_codon_2", "up_aa",
        "down_1nt", "down_2nt", "down_3nt", "down_4nt", "down_5nt", "down_6nt",
        "down_7nt", "down_8nt", "down_123nt", "down_1234nt",
        "aa_gene_position", "nt_CDS_pos", "codon_CDS_pos", "GC%",
        "deltaG_OA", "deltaG_mFold", "tAI_up_1aa", "CAI_up_1aa",
        "dist_to_downstreamstop", "numb_vars_pergene", "Database", "clinical_trial",
        # координаты для будущих join'ов (ClinVar, NMD-скоры):
        "Chromosome", "start", "Ref_allele", "Mutant_allele", "HGVSc", "CLNHGVS",
        "Transcript_ID"]

print("loading treated_samples.rds ...", flush=True)
df = pyreadr.read_r(TOLEDANO)[None][KEEP]
for c in df.columns:
    if str(df[c].dtype) == "category":          # иначе groupby -> OOM (декартово произведение)
        df[c] = df[c].astype(str)
print("raw:", df.shape, flush=True)

# усредняем репликат -> одна строка на (вариант, препарат)
num = df.groupby(["mutation_identifier", "treatment"])[["RT", "sd_RT"]].mean().reset_index()
static = (df.sort_values("RT", na_position="last")
            .groupby("mutation_identifier")[ [c for c in KEEP if c not in ("treatment", "RT", "sd_RT", "mutation_identifier")] ]
            .first().reset_index())
tab = num.merge(static, on="mutation_identifier")
tab.to_parquet("data/model_table.parquet")
print("model_table:", tab.shape,
      "| variants:", tab["mutation_identifier"].nunique(),
      "| treatments:", tab["treatment"].nunique())

# CFTR: измеренная панель (76 вариантов) — внутренний тест-набор
cftr = tab[tab.GENEINFO == "CFTR"]
cftr_wide = cftr.pivot(index="mutation_identifier_short", columns="treatment", values="RT")
cftr_wide.to_csv("data/cftr_measured.csv")
print("cftr_measured:", cftr_wide.shape)

# ClinVar: все нонсенс-варианты CFTR (панель для каталога, ~853)
if not os.path.exists("data/clinvar_cftr_nonsense.tsv"):
    print("downloading ClinVar variant_summary ...", flush=True)
    urllib.request.urlretrieve(CLINVAR_URL, "data/variant_summary.txt.gz")
cv = pd.read_csv("data/variant_summary.txt.gz", sep="\t", low_memory=False)
# nonsense = SNV с белковой нотацией стоп-кодона в Name (p.Gly542Ter / p.G542X / p.*): колонки ProteinChange здесь нет
is_stop = cv.Name.astype(str).str.contains(r"p\.[A-Za-z]{1,3}\d+(Ter|X|\*)", regex=True)
panel = cv[(cv.GeneSymbol == "CFTR") & (cv.Type == "single nucleotide variant") & is_stop].copy()
panel = panel.rename(columns={"#AlleleID": "AlleleID"})
keep_cols = [c for c in ["AlleleID", "Name", "ClinicalSignificance", "ReviewStatus",
                         "Chromosome", "Start", "ReferenceAllele", "AlternateAllele"] if c in panel.columns]
panel[keep_cols].drop_duplicates("AlleleID").to_csv("data/clinvar_cftr_nonsense.tsv", sep="\t", index=False)
print("clinvar_cftr_nonsense:", panel[keep_cols].drop_duplicates("AlleleID").shape)

# TODO(H4, этап 3): join NMD-скоров (NMDetective-AI / predNMD) по Transcript_ID+позиция
# и/или EJC-признак (дистанция PTC до последнего экзон-экзонного соединения).

# QC: пересечение панели с измеренными CFTR-вариантами
meas = set(cftr.mutation_identifier_short)
in_panel = panel.Name.str.extract(r"p\.([A-Za-z]{1,3}\d+)", expand=False).dropna()
print("QC: измеренных CFTR:", len(meas),
      "| ClinVar-панель:", panel.shape[0],
      "| белковых имен в панели распознано:", in_panel.nunique())

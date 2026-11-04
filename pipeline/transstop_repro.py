# Воспроизведение TransStop (Dichopsis/TransStop, NT v2 500M + cross-attention drug head).
# 1) восстановить их данные/сплит (filter.py + data_preparation.py, seed 42);
# 2) найти контекст-победитель пилотом на 400 строках (в репо не сохранён);
# 3) полный пргон их test-сплита (проверка R2=0.94) + их-метрика (expm1, clip 0);
# 4) инференс на CFTR-строках (их train ВИДЕЛ CFTR — случайный сплит; это оценка «памяти», не переноса).
# Выход: results/metrics_transstop.csv, results/summary_transstop.md
import warnings, os, json; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, pyreadr
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from transformers import AutoTokenizer, AutoModelForMaskedLM
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

os.makedirs("TransStop/data", exist_ok=True)
os.makedirs("results", exist_ok=True)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
BASE = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
print("device:", DEV, flush=True)

# ---------- 1) их данные и сплит (детерминированно) ----------
DRUG_ORDER = ["FUr", "Gentamicin", "CC90009", "G418", "Clitocine", "DAP", "SJ6986", "SRI", "Untreated"]
src = pyreadr.read_r("toledano_data/treated_samples.rds")[None]
for c in ["treatment", "viral", "nt_seq", "GENEINFO"]:
    if str(src[c].dtype) == "category":
        src[c] = src[c].astype(str)
frames = []
for name in DRUG_ORDER:
    d = src[(src.treatment == name) & (src.replicate == 2) & (src.reads_allbins > 15)
            & (src.viral == "no") & src.RT.notna()][
        ["nt_seq", "stop_type", "GENEINFO", "mutation_identifier", "Ref_allele", "Mutant_allele", "RT"]]
    frames.append(d.assign(treatment=name))
master = pd.concat(frames, ignore_index=True)
master["nt_seq"] = master.nt_seq.str.upper()

def ctx(seq):  # как их create_contexts: центр 72
    i = 72
    return (seq[i-72:i+75], seq[i-21:i+24], seq[i-9:i+12], seq[i-6:i+9], seq[i-3:i+6], seq[i:i+3])
C = master.nt_seq.apply(ctx)
master[["seq_context_144", "seq_context_42", "seq_context_18", "seq_context_12", "seq_context_6", "seq_context_0"]] = pd.DataFrame(C.tolist(), index=master.index)
master = master.dropna(subset=["seq_context_144"]).reset_index(drop=True)
master["RT_transformed"] = np.log1p(master.RT)

train_val, test_df = train_test_split(master, test_size=0.1, random_state=42, stratify=master["treatment"])
train_df, val_df = train_test_split(train_val, test_size=1/9, random_state=42, stratify=train_val["treatment"])
print(f"их размеры: train {len(train_df)} / val {len(val_df)} / test {len(test_df)}; всего {len(master)}", flush=True)

drug_map = json.loads(open("TransStop/data/drug_map.json").read()) if os.path.exists("TransStop/data/drug_map.json") else \
    json.loads(open(hf_hub_download("Dichopsis/TransStop", "drug_map.json")).read())
master["drug_id"] = master.treatment.map(drug_map)
test_df = master.loc[test_df.index]  # drug_id уже добавлен
cftr_df = master[master.GENEINFO == "CFTR"].copy()
print("CFTR-строк (после их фильтра):", len(cftr_df), flush=True)

# ---------- 2) модель ----------
class PanDrugTransformer(torch.nn.Module):
    def __init__(self, model_name, num_drugs, head_hidden_size, drug_embed_dim, num_attention_heads=8):
        super().__init__()
        full = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        self.base_model = full.base_model
        h = self.base_model.config.hidden_size
        self.drug_embedding = torch.nn.Embedding(num_drugs, drug_embed_dim)
        self.query_projection = torch.nn.Linear(drug_embed_dim, h)
        self.cross_attention = torch.nn.MultiheadAttention(h, num_attention_heads, batch_first=True)
        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(h, head_hidden_size), torch.nn.ReLU(),
            torch.nn.Dropout(0.1), torch.nn.Linear(head_hidden_size, 1))
    def forward(self, input_ids, attention_mask, drug_id):
        seq = self.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        q = self.query_projection(self.drug_embedding(drug_id)).unsqueeze(1)
        att, _ = self.cross_attention(query=q, key=seq, value=seq)
        return self.reg_head(att.squeeze(1)).squeeze(-1)

weights = hf_hub_download("Dichopsis/TransStop", "model.safetensors")
sd = load_file(weights)
head_h = sd["reg_head.0.weight"].shape[0]
drug_d = sd["drug_embedding.weight"].shape[1]
print(f"размерности из state_dict: head {head_h}, drug_embed {drug_d}", flush=True)
model = PanDrugTransformer(BASE, num_drugs=len(drug_map), head_hidden_size=head_h, drug_embed_dim=drug_d)
model.load_state_dict(sd)          # strict=True: контролирует совпадение архитектуры
model.to(DEV).eval()
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

@torch.no_grad()
def predict(df, ctx_col, bs=32):
    preds = []
    for i in range(0, len(df), bs):
        chunk = df.iloc[i:i+bs]
        seqs = [s.replace("U", "T") for s in chunk[ctx_col]]
        enc = tok(seqs, truncation=True, padding="longest", return_tensors="pt")
        ids = enc["input_ids"].to(DEV); mask = enc["attention_mask"].to(DEV)
        drug = torch.tensor(chunk.drug_id.values, dtype=torch.long, device=DEV)
        preds += model(ids, mask, drug).cpu().numpy().tolist()
    return np.array(preds)

def their_r2(y_true_log, y_pred_log):
    t, p = np.expm1(y_true_log), np.expm1(y_pred_log)
    p[p < 0] = 0
    return r2_score(t, p), mean_absolute_error(t, p)

# ---------- 3) пилот: какой контекст обучался ----------
pilot = test_df.sample(min(400, len(test_df)), random_state=0)
pilot_scores = {}
for c in ["seq_context_144", "seq_context_42", "seq_context_18", "seq_context_12", "seq_context_6", "seq_context_0"]:
    p = predict(pilot, c)
    pilot_scores[c] = their_r2(pilot.RT_transformed.values, p)[0]
    print(f"pilot {c}: R2={pilot_scores[c]:.3f}", flush=True)
best_ctx = max(pilot_scores, key=pilot_scores.get)
print("выбран контекст:", best_ctx, flush=True)

# ---------- 4) полный test + CFTR ----------
rows = []
for split, df in [("their_test", test_df), ("cftr_all", cftr_df)]:
    p = predict(df, best_ctx)
    r2, mae = their_r2(df.RT_transformed.values, p)
    rho = spearmanr(df.RT_transformed, p).statistic
    rows.append(dict(split=split, model="transstop_pretrained", drug="ALL",
                     n=len(df), r2_their=r2, mae_their=mae, spearman=rho))
    print(f"{split}: R2={r2:.3f} MAE={mae:.3f} Spearman={rho:.3f} (n={len(df)})", flush=True)
    if split == "cftr_all":
        df = df.copy(); df["pred"] = p
        for drug, g in df.groupby("treatment"):
            if drug == "Untreated":
                continue
            rows.append(dict(split="cftr_perdrug", model="transstop_pretrained", drug=drug,
                             n=len(g), r2_their=np.nan, mae_their=np.nan,
                             spearman=spearmanr(g.RT_transformed, g.pred).statistic))
        # top-1 выбора лучшего препарата
        wide = df.pivot_table(index="mutation_identifier", columns="treatment", values=["RT", "pred"])
        drugs = [d for d in DRUG_ORDER if d != "Untreated"]
        t = wide["RT"][drugs]; p_w = wide["pred"][drugs]
        ok = t.notna().sum(1) == len(drugs)
        top1 = float((p_w[ok].values.argmax(1) == t[ok].values.argmax(1)).mean())
        rows.append(dict(split="cftr_topdrug", model="transstop_pretrained", drug="top1",
                         n=int(ok.sum()), r2_their=np.nan, mae_their=np.nan, spearman=top1))
        print(f"CFTR top-1 выбора препарата: {top1:.3f} (n={int(ok.sum())})", flush=True)

res = pd.DataFrame(rows)
res.to_csv("results/metrics_transstop.csv", index=False)
with open("results/summary_transstop.md", "w") as f:
    f.write("# TransStop (pretrained) воспроизведение\n\n")
    f.write(f"Контекст-победитель (пилот): {best_ctx}\n\n")
    f.write(res.to_string(index=False))
    f.write("\n\nКонтекст: их test-сплит — модель видела CFTR в train (случайный сплит); "
            "cftr_* — оценка памяти модели, НЕ переноса. Наш cftr_holdout (etrain без CFTR): "
            "enet_tuned 0.614 / mt_hgb 0.621 / hgb 0.630 Spearman; top-1 0.592.\n")
print("\nготово: results/metrics_transstop.csv")

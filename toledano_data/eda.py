# EDA for Toledano et al. 2024 readthrough dataset (treated_samples.rds)
# Output: figs/*.png + cftr_readthrough.csv + stdout stats
import warnings
warnings.filterwarnings("ignore")
import pyreadr, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# palette (dataviz reference, light mode)
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})
seq_cmap = LinearSegmentedColormap.from_list("seq", SEQ)

df = pyreadr.read_r("treated_samples.rds")[None]
df = df[["mutation_identifier", "GENEINFO", "mutation_identifier_short",
         "treatment", "RT", "stop_type", "down_1nt"]]  # ponytail: drop 137 string cols to keep RAM ~100MB
for c in df.columns:  # categorical -> str, else groupby explodes into full category cross-product (OOM)
    if str(df[c].dtype) == "category":
        df[c] = df[c].astype(str)
print(f"loaded {df.shape}", flush=True)

# ---- tidy: average replicates -> one RT per (variant, treatment) ----
tidy = (df.groupby(["mutation_identifier", "GENEINFO", "mutation_identifier_short",
                    "treatment"], as_index=False)["RT"].mean())
wide = tidy.pivot(index="mutation_identifier", columns="treatment", values="RT")
print(f"variants: {wide.shape[0]} | treatments: {wide.shape[1]}")
print("missing RT per treatment:",
      tidy.groupby("treatment")["RT"].apply(lambda s: s.isna().sum()).to_dict())

DRUGS = [c for c in wide.columns if c != "Untreated"]
order = wide[DRUGS].median().sort_values(ascending=False).index.tolist()

# ---- Fig 1: RT distribution per drug (log y, single hue) ----
fig, ax = plt.subplots(figsize=(8, 4.2))
data = [wide[d].dropna().values for d in order]
bp = ax.boxplot(data, tick_labels=order, showfliers=False, patch_artist=True,
                medianprops=dict(color=INK, lw=1.5), widths=.55)
for b in bp["boxes"]:
    b.set(facecolor=C1, edgecolor=C1, alpha=.55)
ax.plot([], [])  # noop
ax.set_yscale("log")
ax.set_ylabel("Readthrough, % от WT (log)")
ax.set_title("Распределение readthrough по препаратам (медиана, 25–75 перц.)")
ax.grid(axis="y", color="#e5e4e0", lw=.7)
ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figs/fig1_rt_distribution.png", dpi=150); plt.close(fig)

# ---- Fig 2: stop-type effect ----
st = (df.groupby(["mutation_identifier", "treatment", "stop_type"], as_index=False)["RT"].mean())
med = st.groupby(["treatment", "stop_type"])["RT"].median().unstack()[["uga", "uag", "uaa"]]
med = med.loc[order]
x = np.arange(len(med)); w = .26
fig, ax = plt.subplots(figsize=(8, 4.2))
for i, (col, color) in enumerate(zip(["uga", "uag", "uaa"], [C1, C2, C3])):
    ax.bar(x + (i - 1) * w, med[col], w, color=color, label={"uga": "UGA", "uag": "UAG", "uaa": "UAA"}[col])
ax.set_xticks(x, med.index, rotation=30, ha="right")
ax.set_ylabel("Медианный readthrough, %")
ax.set_title("Эффект типа стоп-кодона по препаратам")
ax.legend(frameon=False)
ax.grid(axis="y", color="#e5e4e0", lw=.7); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figs/fig2_stoptype.png", dpi=150); plt.close(fig)

# ---- Fig 3: +4 nucleotide effect (heatmap) ----
p4 = (df.groupby(["mutation_identifier", "treatment", "down_1nt"], as_index=False)["RT"].mean())
h = p4.groupby(["down_1nt", "treatment"])["RT"].median().unstack().loc[["a", "u", "g", "c"], order]
fig, ax = plt.subplots(figsize=(8, 2.8))
im = ax.imshow(h.values, cmap=seq_cmap, aspect="auto")
ax.set_xticks(range(len(order)), order, rotation=30, ha="right")
ax.set_yticks(range(4), [b.upper() for b in h.index])
for i in range(4):
    for j in range(len(order)):
        v = h.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="#ffffff" if v > np.nanmax(h.values) * .6 else INK, fontsize=8)
ax.set_title("Медианный readthrough по нуклеотиду +4 (после стоп-кодона)")
fig.colorbar(im, ax=ax, shrink=.8, label="%")
fig.tight_layout(); fig.savefig("figs/fig3_plus4nt.png", dpi=150); plt.close(fig)

# ---- Fig 4: drug-drug Spearman correlation ----
corr = wide[DRUGS].corr("spearman")
fig, ax = plt.subplots(figsize=(6.4, 5.2))
im = ax.imshow(corr.values, cmap=DIV, vmin=0.3, vmax=1)
ax.set_xticks(range(len(DRUGS)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(DRUGS)), corr.columns)
for i in range(len(DRUGS)):
    for j in range(len(DRUGS)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="#ffffff" if (corr.values[i, j] > .85 or corr.values[i, j] < .45) else INK)
ax.set_title("Корреляция Спирмена препаратов между собой")
fig.colorbar(im, ax=ax, shrink=.8)
fig.tight_layout(); fig.savefig("figs/fig4_drug_corr.png", dpi=150); plt.close(fig)

# ---- best-drug analysis (variant level) ----
best = wide[DRUGS].max(axis=1)
for thr in [0.5, 1.0, 2.0]:
    print(f"RT > {thr}%: any drug {100 * (best > thr).mean():.1f}% | "
          f"DAP {100 * (wide['DAP'] > thr).mean():.1f}% | "
          f"Gentamicin {100 * (wide['Gentamicin'] > thr).mean():.1f}%")

# ---- CFTR ----
cf = tidy[tidy["GENEINFO"] == "CFTR"]
cfw = cf.pivot(index="mutation_identifier_short", columns="treatment", values="RT")
cfw = cfw.loc[cfw[DRUGS].median(axis=1).sort_values(ascending=False).index]
cfw.round(3).to_csv("cftr_readthrough.csv")
print(f"\nCFTR variants measured: {cfw.shape[0]} (ClinVar всего 853 nonsense)")
print("top-10 CFTR по медианному RT:\n", cfw.head(10).round(2).to_string())

# ---- Fig 5: CFTR heatmap, top 25 ----
top = cfw.head(25)
fig, ax = plt.subplots(figsize=(8, 7.4))
v = top[[*order, "Untreated"]].values.astype(float)
v = np.ma.masked_invalid(v)
im = ax.imshow(v, cmap=seq_cmap, aspect="auto", vmin=0, vmax=np.nanmax(v))
ax.set_xticks(range(v.shape[1]), [*order, "Untreated"], rotation=45, ha="right")
ax.set_yticks(range(len(top)), top.index)
for j, name in enumerate(top.index):
    row = v[j]
    k = np.nanargmax(row.filled(-np.inf))
    ax.text(k, j, "•", ha="center", va="center", color=INK, fontsize=11)
ax.set_title("CFTR: readthrough по препаратам (топ-25 вариантов, • = лучший препарат)")
fig.colorbar(im, ax=ax, shrink=.7, label="%")
fig.tight_layout(); fig.savefig("figs/fig5_cftr.png", dpi=150); plt.close(fig)

# ---- sanity checks ----
assert cfw.shape[0] == cf["mutation_identifier_short"].nunique()
assert wide.shape[1] == 9
print("\nOK")

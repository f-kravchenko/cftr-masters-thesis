# Схема принципа readthrough-терапии (для презентации, п.2 пожеланий научрука)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

S, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
C_MRNA, C_STOP, C_DRUG, C_OK, C_BAD = "#2a78d6", "#e34948", "#eb6834", "#1baf7a", "#a8a7a2"
plt.rcParams.update({"figure.facecolor": S, "axes.facecolor": S, "savefig.facecolor": S,
                     "text.color": INK, "font.size": 11})
fig, ax = plt.subplots(figsize=(10.6, 3.9))
ax.set_xlim(0, 10.6); ax.set_ylim(0, 3.9); ax.axis("off")

def mrna(y, x0, x1, stop_at, n=16):
    xs = __import__("numpy").linspace(x0, x1, n)
    for i, x in enumerate(xs):
        c = C_STOP if i == stop_at else C_MRNA
        r = 0.115
        ax.add_patch(Circle((x, y), r, fc="none", ec=c, lw=1.8, zorder=3))
    return xs

def ribosome(x, y, label="рибосома"):
    ax.add_patch(FancyBboxPatch((x-0.52, y-0.38), 1.04, 0.76,
                 boxstyle="round,pad=0.02,rounding_size=0.18", fc="#eef1f6", ec=INK2, lw=1.2, zorder=2))
    ax.text(x, y, label, ha="center", va="center", fontsize=9.5, color=INK)

def protein(x0, y, x1, ok, label):
    c = C_OK if ok else C_BAD
    ax.plot([x0, x1], [y, y], lw=6, color=c, solid_capstyle="round")
    ax.text((x0+x1)/2, y+0.22, label, ha="center", fontsize=9.5,
            color=INK if ok else INK2)

def arrow(x0, y0, x1, y1, color=INK2, style="-|>", lw=1.4, text=None, tdx=0, tdy=0.12):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                 mutation_scale=13, color=color, lw=lw))
    if text:
        ax.text((x0+x1)/2+tdx, (y0+y1)/2+tdy, text, ha="center", fontsize=9, color=color)

# --- мРНК с PTC ---
xs = mrna(2.55, 0.55, 4.05, stop_at=8)
ax.text(0.55, 3.35, "мРНК CFTR с нонсенс-мутацией", fontsize=11, weight="bold")
ax.annotate("преждевременный\nстоп-кодон (PTC)", xy=(xs[8], 2.55+0.12), xytext=(xs[8]-0.65, 3.55),
            fontsize=9, color=C_STOP, ha="center",
            arrowprops=dict(arrowstyle="->", color=C_STOP, lw=1.2))
ribosome(1.75, 2.55)

# --- развилка ---
arrow(2.45, 2.55, 4.6, 3.1, color=C_BAD, text="терминация\n(как обычно)", tdx=-0.35, tdy=0.18)
arrow(2.45, 2.55, 4.6, 1.55, color=C_OK, text="readthrough\n(редко)", tdx=-0.32, tdy=-0.10)

# --- левая ветка: без препарата ---
ax.text(5.0, 3.62, "Без препарата", fontsize=10.5, weight="bold", color=INK2, ha="center")
protein(5.25, 3.15, 6.45, ok=False, label="обрезанный\nнефункциональный белок")
ax.text(6.75, 3.15, "+ NMD: мРНК\nраспадается", fontsize=9, color=INK2, va="center")

# --- правая ветка: с препаратом ---
ax.text(5.0, 2.02, "С readthrough-препаратом", fontsize=10.5, weight="bold", color=C_DRUG, ha="center")
ax.add_patch(Circle((4.35, 1.5), 0.14, fc=C_DRUG, ec="none"))
ax.annotate("препарат помогает рибосоме\n«проскочить» стоп-кодон", xy=(4.5, 1.5), xytext=(3.3, 0.75),
            fontsize=9, color=C_DRUG, arrowprops=dict(arrowstyle="->", color=C_DRUG, lw=1.2))
protein(5.25, 1.5, 8.6, ok=True, label="полноформатный CFTR — терапевтический эффект")
ax.text(6.9, 1.02, "вставляется другая аминокислота → работает ли белок, зависит от контекста",
        fontsize=8.5, color=INK2, ha="center", style="italic")

fig.tight_layout()
fig.savefig("results/pres/fig_scheme.png", dpi=200)
print("схема: results/pres/fig_scheme.png")

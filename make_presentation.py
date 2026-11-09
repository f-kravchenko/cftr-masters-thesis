# Заполнение шаблона презентации (ОНИТ) — версия 2 после встречи с научруком:
# порядок: титул → биоконтекст (схема) → анализ источников → вопрос → проблема/новизна →
# проблема/актуальность → цель → гипотезы (1+2, с критериями) → методология → методика →
# данные → препараты → результаты → источники → спасибо.
import copy, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation

# ---------------- графики (палитра dataviz, light) ----------------
S, INK, INK2, C1, C2, C3 = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": S, "axes.facecolor": S, "savefig.facecolor": S,
    "text.color": INK, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False})
os.makedirs("results/pres", exist_ok=True)

def style(ax):
    ax.grid(axis="y", color="#e5e4e0", lw=.7); ax.set_axisbelow(True)

fig, ax = plt.subplots(figsize=(5.2, 3.2))
labels = ["CFTR-only\n(76 вар.)", "CFTR\nполностью в тесте", "невидимые\nгены", "случайный\nсплит"]
vals = [0.361, 0.630, 0.704, 0.705]
ax.bar(labels, vals, color=C1, width=.62)
for i, v in enumerate(vals):
    ax.text(i, v + .015, f"{v:.3f}", ha="center", fontsize=10)
ax.axhline(0.881, color=C2, lw=1.6)
ax.text(3.45, 0.893, "шумовой потолок 0.881", color=C2, fontsize=9, ha="right")
ax.set_ylim(0, 1.0); ax.set_ylabel("Spearman ρ"); style(ax)
ax.set_title("Перенос модели (бустинг, 8 препаратов)", fontsize=11)
fig.tight_layout(); fig.savefig("results/pres/fig_splits.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 3.2))
labels = ["без признака\nпрепарата", "+ профиль\nпрепарата", "потолок режима\n(7 измерений)"]
vals = [0.474, 0.518, 0.628]
ax.bar(labels, vals, color=[C3, C1, "#c9c8c3"], width=.58)
for i, v in enumerate(vals):
    ax.text(i, v + .015, f"{v:.3f}", ha="center", fontsize=10)
ax.annotate("82% потолка\nбез измерений варианта", xy=(1, 0.53), xytext=(1.6, 0.30),
            fontsize=9, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2))
ax.set_ylim(0, 0.75); ax.set_ylabel("Spearman ρ"); style(ax)
ax.set_title("Невидимый препарат (leave-one-drug-out)", fontsize=11)
fig.tight_layout(); fig.savefig("results/pres/fig_lodo.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 3.2))
labels = ["вообще без модели:\nперенос RT того же\nварианта с 8 препаратов", "TransStop\n(NT v2 500M,\n drug-эмбеддинги)"]
vals = [0.863, 0.940]
ax.bar(labels, vals, color=["#c9c8c3", C2], width=.45)
for i, v in enumerate(vals):
    ax.text(i, v + .008, f"{v:.3f}", ha="center", fontsize=10)
ax.set_ylim(0, 1.05); ax.set_ylabel("R² (их метрика)"); style(ax)
ax.set_title("Их случайный сплит: 0.86 из 0.94 — без модели", fontsize=11)
fig.tight_layout(); fig.savefig("results/pres/fig_transstop.png", dpi=200); plt.close(fig)
print("графики готовы", flush=True)

# ---------------- заполнение pptx ----------------
SRC = "Шаблон Финальная презентация концепции исследования (ОНИТ, 2025_2026).pptx"
DST = "Презентация концепции — Кравченко (ОНИТ, 2025_2026).pptx"
prs = Presentation(SRC)

def set_text(tf, text):
    lines = text.split("\n")
    p0 = tf.paragraphs[0]
    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)
    if not p0.runs:
        p0.add_run().text = lines[0]
    else:
        for r in list(p0.runs)[1:]:
            r._r.getparent().remove(r._r)
        p0.runs[0].text = lines[0]
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    for line in lines[1:]:
        newp = copy.deepcopy(p0._p)
        tf._txBody.append(newp)
        runs = newp.findall("a:r", ns)
        for extra in runs[1:]:
            newp.remove(extra)
        if runs:
            runs[0].find("a:t", ns).text = line

def fill(slide_idx, replacements, pictures=None):
    s = prs.slides[slide_idx]
    for sh in s.shapes:
        if not sh.has_text_frame:
            continue
        cur = " ".join(r.text for p in sh.text_frame.paragraphs for r in p.runs)
        cur_norm = " ".join(cur.split())
        for key, txt in replacements:
            if " ".join(key.split()) in cur_norm:
                set_text(sh.text_frame, txt)
                replacements = [(k, t) for k, t in replacements if k != key]
                break
    for path, l, t, w in (pictures or []):
        s.shapes.add_picture(path, int(l * 914400), int(t * 914400), width=int(w * 914400))
    assert not replacements, f"слайд {slide_idx}: не найдены якоря {[k for k, _ in replacements]}"

# 1: титул
fill(1, [
    ("Выполнил", "Выполнил: Кравченко Фёдор\n\nНаучный руководитель: ____"),
    ("Тема исследования",
     "In silico-стратификация нонсенс-вариантов CFTR\nдля readthrough-терапии: drug-aware предсказание\nпо последовательности"),
    ("Название магистерской программы", "Магистерская программа: «Науки о данных» (ОНИТ)"),
])

# 19 (заготовка): биологический контекст + схема
fill(19, [
    ("Заголовок слайда", "Контекст: от нонсенс-мутации к задаче ML"),
    ("Изображение", ""),
    ("Подпись к изображению", "Принцип readthrough-терапии (схема автора)"),
    ("Текст Текст", "• Муковисцидоз — тяжёлое генетическое заболевание; причина — мутации гена CFTR\n"
                    "• У ~10% пациентов мутации — нонсенс: преждевременный стоп-кодон (PTC) → белок не синтезируется, модуляторы им недоступны в принципе\n"
                    "• Readthrough-терапия: препарат помогает рибосоме «проскочить» стоп → синтез продолжается\n"
                    "• Эффект зависит от пары «мутация × препарат» и варьирует на порядки; проверка каждой пары (репортеры, органоиды) дорога и не масштабируется\n"
                    "• DS-задача: не заменить эксперимент, а приоритизировать пары для проверки"),
], pictures=[("results/pres/fig_scheme.png", 6.1, 0.55, 3.5)])

# 2: анализ источников
fill(2, [
    ("Охарактеризуйте выводы",
     "25 источников (2023–2026), полный разбор — в репозитории (literature/, 26 файлов):\n"
     "• Toledano 2024: RT определяется локальным контекстом; подбор препарата удваивает долю отвечающих\n"
     "• TransStop 2025 (preprint): drug-aware трансформер, R²=0.94 — но только случайный сплит\n"
     "• RT и NMD-оси измерены по отдельности, не интегрированы никем\n"
     "• Клиника: ataluren и ELX-02 провалились без стратификации пациентов\n"
     "Из этих пробелов вытекает исследовательский вопрос и гипотезы"),
    ("Ссылка на\nполный анализ",
     "github.com/f-kravchenko/cftr-masters-thesis — literature/"),
])

# 3: исследовательский вопрос
fill(3, [
    ("Кратко сформулируйте",
     "Насколько точно ответ пары «нонсенс-вариант × readthrough-препарат» предсказуем из последовательности и формализуемых свойств препарата — и переносится ли предсказание на варианты, гены и препараты, отсутствовавшие в обучении?"),
])

# 4: проблема (научная) и новизна
fill(4, [
    ("Опишите \nтеоретический компонент проблемы",
     "Научная проблема: опубликованные readthrough-модели валидируются на случайных сплитах и предсказывают только репортерный RT — переносимость на новые варианты, гены и препараты не доказана.\n"
     "Новизна: первая систематическая оценка обобщающей способности readthrough-моделей (честные сплиты, мажоритарный/CFTR-only/model-free бейзлайны, шумовой потолок) + протокол приоритизации пар для CFTR, валидированный на трёх независимых системах. Показано: 0.65–0.86 из R²=0.94 SOTA — артефакт сплита."),
])

# 5: проблема (практическая) и актуальность
fill(5, [
    ("Опишите \nпрактический компонент проблемы",
     "Практическая проблема: пары «мутация × препарат» нельзя перебрать экспериментально — репортеры и органоиды дороги и плохо масштабируются; без дешёвого приоритизатора новая волна readthrough-терапий повторит провал ataluren.\n"
     "Актуальность (масштаб): ~10% пациентов МВ — ≈5–7 тыс. в Европе, ≈400–450 в РФ — не имеют терапии принципиально (нет белка-мишени для модуляторов); следующей волнe препаратов при входе в клинику нужен инструмент отбора когорт."),
])

# 6: цель
fill(6, [
    ("Сформулированная в 1", "Цель: разработать и валидировать drug-aware модель приоритизации пар «нонсенс-вариант CFTR × readthrough-препарат» с честной оценкой переноса на невидимые варианты, гены и препараты."),
    ("Объект исследования", "Объект: эффективность трансляционного readthrough патогенных преждевременных стоп-кодонов.\nПредмет: закономерности «последовательность + свойства препарата → RT» и их переносимость."),
])

# 7: гипотезы (1 главная + 2 частные, фальсифицируемые)
fill(7, [
    ("На основе теоретического анализа",
     "Главная H: ответ пары (вариант, препарат) предсказуем из локального контекста PTC и признаков препарата и переносится на невидимое.\n"
     "Критерии a priori: (а) CFTR-holdout Spearman ≥ 0.55 И top-1 выбора препарата ≥ 55%; (б) few-shot LODO ≥ 75% потолка (≥0.47). Провал любого порога = опровержение. Статус: (а) 0.62–0.64 и 59% ✓; (б) 0.518 ✓\n"
     "Частная H1 (композиция): для эндогенного исхода RT недостаточно — количественная NMD-ось добавляет ≥ +0.2 Spearman к согласованию с органоидами; прирост < +0.1 = опровержение. Статус: ждём скоры NMDetective-AI (GPU).\n"
     "Частная H2 (zero-shot): молекулярные дескрипторы препарата дают zero-shot LODO ≥ 0.8× профильного (≥0.42). Проверка: курэйтинг SMILES из супплемента Toledano."),
    ("Текст текст текст", ""),
])

# 8: методология
fill(8, [
    ("Методология должна быть описана",
     "Методология: прикладное машинное обучение на открытых биомедицинских данных (sort-seq/DMS) в парадигме честной оценки обобщения.\nМетодика: (1) сборка данных; (2) лестница бейзлайнов до воспроизведения SOTA; (3) сплиты без утечек; (4) внешняя кросс-системная валидация; (5) критический аудит всех метрик."),
    ("Текст текст текст",
     "Данные: 5 485 вариантов × 9 условий × 2 реплики (Toledano 2024); внешние: dual-luciferase 13 CFTR-PTC (Mangkalaphiban 2024), 206 органоидов × ELX-02 (Destoop 2026)."),
])

# 9: методика
fill(9, [
    ("Методика описывается пошагово",
     "Выполнено: (1) сборка model_table 5 485×9 + ClinVar-панель; (2) бейзлайны: правила → elastic net (тюнинг) → бустинг; (3) сплиты random5 / genegroup5 / CFTR-holdout / LODO; (4) воспроизведение TransStop из открытых весов (R²=0.940 подтверждён) + model-free бейзлайн; (5) NMD-признаки из GENCODE v47; (6) внешняя валидация Mangkalaphiban + Destoop; (7) аудит: фикс утечки LODO, мажоритарные/CFTR-only бейзлайны, шумовой потолок 0.881."),
    ("Ссылка на\nдетальное описание",
     "Пошаговые методики и код: github.com/f-kravchenko/cftr-masters-thesis\n(10 чекпоинтов, pipeline/, results/)"),
])

# 10: источники данных
fill(10, [
    ("Используемые для исследования данные",
     "• Toledano 2024 (figshare/SRA): 5 485 PTC × 8 препаратов — обучение и сплиты\n"
     "• Mangkalaphiban 2024 (Zenodo): 13 CFTR-PTC, dual-luciferase — внешняя валидация\n"
     "• Destoop 2026 (bioRxiv): 206 органоидов × ELX-02 — эндогенная валидация\n"
     "• ClinVar: панель нонсенс-вариантов CFTR\n"
     "• GENCODE v47: NMD-признаки\n"
     "• TransStop (GitHub/HF): воспроизведение SOTA\nВсе данные открытые; сборка — pipeline/build_dataset.py"),
])

# 16 (заготовка): препараты
fill(16, [
    ("Заголовок слайда", "Препараты: четыре механизма действия"),
    ("Текст Текст",
     "• Аминогликозиды (рибосома, декодирующий центр): гентамицин, G418, ELX-02\n"
     "• Нуклеозидные аналоги (встраиваются в мРНК): clitocine, 5-фторуридин\n"
     "• DAP = 2,6-диаминопурин (модификация тРНК через FTSJ1; UGA-специфичен)\n"
     "• Деградеры факторов терминации: SRI-41315 (eRF1); SJ6986, CC-90009 (eRF3a/GSPT1)\n\n"
     "Для ML: объединять классы корректно только с признаками препарата (корреляции RT: 0.93 внутри механизма против 0.39 между).\n"
     "Свойства препарата: MOA-класс (4) + мишень + молекулярные дескрипторы (SMILES → RDKit) + профиль сродства.\n"
     "Конфаунды: clitocine стабилизирует РНК; CC-90009 подавляет NMD; ataluren вне данных, мишень не установлена."),
])

# 11: результаты
fill(11, [
    ("План реализации", "Результаты: что сделано"),
    ("На этом слайде вы отражаете", "После критического аудита всех метрик:"),
    ("…", "• Перенос на CFTR: Spearman 0.62–0.64 без единого CFTR-варианта в train (CFTR-only — 0.36)\n"
           "• Выбор лучшего препарата: top-1 59% (мажоритарный бейзлайн 43%), top-2 80% (59%)\n"
           "• Невидимый препарат: 0.52 = 82% информационного потолка без измерений варианта\n"
           "• Внешняя валидация: 0.555 при межсистемном потолке 0.786; RT не предсказывает органоидный ответ (−0.22) → нужны NMD- и белковые оси"),
], pictures=[
    ("results/pres/fig_splits.png", 0.45, 2.55, 3.05),
    ("results/pres/fig_lodo.png", 3.55, 2.55, 3.05),
    ("results/pres/fig_transstop.png", 6.65, 2.55, 3.05),
])

# 12-13: источники
fill(12, [
    ("Список", "1. Toledano I., Supek F., Lehner B. Genome-scale quantification and prediction of pathogenic stop codon readthrough by small molecules // Nature Genetics. 2024. 56(9):1914–1924.\n"
                "2. TransStop. bioRxiv 2025.08.29.672857 (preprint).\n"
                "3. Mangkalaphiban A. et al. // Nature Communications. 2024. 15:2486.\n"
                "4. Destoop E. et al. // J. Cystic Fibrosis. 2026; bioRxiv 2026.07.31.740556.\n"
                "5. Veiner I. et al. NMDetective-AI. bioRxiv 2026.03.24.714003."),
])
fill(13, [
    ("Список", "6. Linde L. et al. // J. Clin. Invest. 2007. 117(3):683–692.\n"
                "7. Konstan M. et al. // Lancet Respir Med. 2014; J. Cyst. Fibros. 2020 (ataluren, фазы 3).\n"
                "8. Thrasher S. et al. // J. Cystic Fibrosis. 2026 (вставка аминокислот, LC-MS/MS).\n"
                "9. McDonald C. et al. // PLOS ONE. 2024.\n"
                "10. Полный обзор: literature/ (26 файлов) в репозитории проекта."),
])

# 14: спасибо
fill(14, [("Спасибо", "Спасибо за внимание!\nРепозиторий: github.com/f-kravchenko/cftr-masters-thesis\nВопросы?")])

# ---- порядок слайдов: титул, контекст(19), источники(2), вопрос(3), пробл/новизна(4),
# пробл/актуальность(5), цель(6), гипотезы(7), методология(8), методика(9), данные(10),
# препараты(16), результаты(11), литература(12,13), спасибо(14), остальные заготовки.
order = [1, 19, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 11, 12, 13, 14] + [i for i in range(17, 30) if i not in (19,)]
xml_slides = prs.slides._sldIdLst
slides = list(xml_slides)
for el in slides:
    xml_slides.remove(el)
for i in order:
    xml_slides.append(slides[i])

prs.save(DST)
print("готово:", DST, "| слайдов:", len(prs.slides.__iter__.__self__._sldIdLst))

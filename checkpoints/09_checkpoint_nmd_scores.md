# Чекпоинт 9. Попытка количественных NMD-скоров NMDetective-AI (04.11.2026)

Артефакты: `data/ucsc/nmdDetectAi.{bb,bw}` (165+75 МБ, hgdownload UCSC), `pipeline/nmd_ai_scores.py` (готов к запуску после получения скоров).

## Результат проверки готовых скоров — отрицательный

1. **Стек модели на маке не ставится**: требует CUDA-only пакеты (`mamba-ssm`, `causal-conv1d`), README — «Linux + NVIDIA GPU».
2. **UCSC-трек сломан для наших целей** (нативный bigBed/bigWig `nmdDetectiveAi`, hg38):
   - CFTR: всего **5 записей** (кодоны 2–14) с координатами **вне гена** (~117.48 Мб против реального спана 117.12–117.31 Мб) и цепью `+` у минус-гена — следствие бага их генерации;
   - на bigWig сигнала на реальных координатах CFTR нет (TP53 — есть);
   - покрытие панели: **4 из 20** случайных генов имеют записи у позиции варианта.
   → использовать трек нельзя: для CFTR данных нет вовсе, для остальных — неотличимо, где ещё баг.
3. Скоры из репозитория предвычисленными таблицами не поставляются; predNMD (Berkeley) bulk-выгрузок не публикует.

## Решение

Количественная ось NMD = **самостоятельный прогон NMDetective-AI на GPU** (GTX 1660 подходит: для инференса достаточно, стек README-протестирован под CUDA). План запуска на Linux-машине:

```
git clone https://github.com/Vejni/NMDetectiveAI && cd NMDetectiveAI
git lfs install && git lfs pull                 # веса models/NMDetectiveAI.pt
conda create -y -n nmd python=3.10 pip && conda activate nmd
conda install -y -c conda-forge -c bioconda -c pytorch numpy pandas scipy pytorch genomekit loguru typer
pip install "transformers==4.50.3" "tokenizers==0.21.1" "huggingface-hub==0.30.1"
conda install conda-forge::causal-conv1d conda-forge::mamba-ssm
pip install -e .
# CFTR (smoke-тест), затем цикл по генам панели (~2 000):
python main.py predict plot-transcript-predictions --gene-name CFTR \
    --model-path models/NMDetectiveAI.pt --save-predictions
```

Выход — per-транскрипт CSV (`ptc_position`, `prediction`); джойн к нам по (ENST, кодон) — код в `pipeline/nmd_ai_scores.py` уже написан (после прогона заменить источник скоров).

## Состояние H4

Механистические признаки (этап 3) — отрицательный контроль пройден. Количественная проверка (Destoop-композит) **отложена до прогона модели на GPU** — без неё H4 остаётся открытой с доказанной необходимостью количественной оси (бинарное правило без дисперсии, чекпоинт 08).

## Попутно

Проверка трека — самостоятельный мини-результат: опубликованный UCSC-ресурс NMDetective-AI непригоден для per-вариантных join'ов (баги координат, неполное насыщение). В тексте — как обоснование самостоятельного прогона.

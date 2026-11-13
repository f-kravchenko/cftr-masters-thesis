# ELX-02 (NB124 → exaluren) — глубокий обзор: химия, преклиника, клиника, статус

**Статус:** сводка по 10+ первоисточникам, все ссылки верифицированы (ноябрь 2026) · **Роль для нас:** единственный readthrough-препарат с современной клинической программой; отсутствует в обучающих данных Toledano → главный кандидат на zero/few-shot предсказание; его органоидные датасеты (Destoop, Crawford) — внешняя валидация.

## Библиография (для списка литературы)

1. Kandasamy J., Atia-Glikin D., Shulman E., et al. Increased selectivity towards cytoplasmic versus mitochondrial ribosome confers improved efficiency of synthetic aminoglycosides in fixing damaged genes // J. Med. Chem. 2012. Vol. 55, № 23. P. 10630–10643. doi:10.1021/jm3012992. [PMC3542475](https://pmc.ncbi.nlm.nih.gov/articles/PMC3542475) — дизайн серии NB, соединение (S)-11 = NB124.
2. Shulman E., Belakhov V., Wei G., et al. Designer aminoglycosides that selectively inhibit cytoplasmic rather than mitochondrial ribosomes show decreased ototoxicity // J. Biol. Chem. 2014. 289(4):2318–2330. doi:10.1074/jbc.M113.533588. [PMC3900975](https://pmc.ncbi.nlm.nih.gov/articles/PMC3900975).
3. Xue X., Mutyam V., Tang L., et al., Bedwell D.M., Rowe S.M., Baasov T. Synthetic aminoglycosides efficiently suppress CFTR nonsense mutations and are enhanced by ivacaftor // Am. J. Respir. Cell Mol. Biol. 2014. 50(3):614–622. doi:10.1165/rcmb.2013-0282OC. [PMC4068923](https://pmc.ncbi.nlm.nih.gov/articles/PMC4068923) — базовая преклиника CF.
4. Crawford D.K., et al. Targeting G542X CFTR nonsense alleles with ELX-02 restores CFTR function in human-derived intestinal organoids // J. Cyst. Fibros. 2021. 20(3):534–541. doi:10.1016/j.jcf.2021.01.009. [PMID 33558100](https://pubmed.ncbi.nlm.nih.gov/33558100).
5. de Poel E., et al. Functional restoration of CFTR nonsense mutations in intestinal organoids // J. Cyst. Fibros. 2022. 21(2):e34–e41. doi:10.1016/j.jcf.2021.09.020. [PMID 34666947](https://pubmed.ncbi.nlm.nih.gov/34666947).
6. Pranke I.M., et al., Bidou L., Namy O., Sermet-Gaudelus I. The U UGA C sequence provides a favorable context to ELX-02 induced CFTR readthrough // J. Cyst. Fibros. 2023. 22(3):560–563. doi:10.1016/j.jcf.2022.10.010. [PMID 36400713](https://pubmed.ncbi.nlm.nih.gov/36400713).
7. Chen J., et al., Bedwell D.M., Rowe S.M. ELX-02 induces readthrough of G550X-CFTR producing superfunctional protein... // Am. J. Physiol. Lung. 2023. 324(6):L756–770. doi:10.1152/ajplung.00038.2023. [PMC10202470](https://pmc.ncbi.nlm.nih.gov/articles/PMC10202470).
8. Gale D.P., Böckenhauer D., et al. Phase 2a study of readthrough therapy in nonsense mutation Alport syndrome // J. Clin. Invest. 2026 (online 12.11.2026). doi:10.1172/JCI208623. [PMID 43019013](https://pubmed.ncbi.nlm.nih.gov/43019013/) — первое опубликованное in vivo доказательство readthrough у exaluren.
9. Borrelli A., et al. (Galietta) // Eur. Respir. J. 2026. doi:10.1183/13993003.01171-2025. [PMID 41381228](https://pubmed.ncbi.nlm.nih.gov/41381228) — воспаление усиливает спасение G542X/W1282X.
10. Popadynec M., et al. // ACS Med. Chem. Lett. 2021. doi:10.1021/acsmedchemlett.1c00349. [PMC8436412](https://pmc.ncbi.nlm.nih.gov/articles/PMC8436412) — дальнейшая SAR-оптимизация.

## Что сделано (по блокам)

**Химия (1–3, 10).** Синтетический псевдотрисахаридный аминогликозид серии NB лаборатории Баасова (Technion): паромициновый остов + три фармакофора (N1-AHB, 6′-(R)-Me, 5″-(S)-Me). Ключевая идея дизайна: активность readthrough коррелирует с ингибированием цитоплазматической рибосомы (IC50 0.7 мкМ, в ~88 раз сильнее гентамицина), токсичность — митохондриальной (NB124 в ~38 раз слабее G418 на миторибосомах); снижение ототоксичности показано отдельно.

**Преклиника CF (3–7, 9).** Xue 2014: сильнейший индуктор readthrough G542X/R1162X/W1282X, восстановление хлорного транспорта в первичных HBE до ~7% WT, синергия с ивакафтором. Серия на пациент-производных моделях: кишечные органоиды (Crawford 2021, G542X), RT×NMDi×модулятор (de Poel 2022), влияние контекста UUGAC (Pranke 2023 — «ручной» предшественник нашей контекстной модели), G550X с суперфункциональным белком и единственной вставкой Trp (Chen 2023), усиление при воспалении (Borrelli 2026).

**Клиника CF.** Фазы 1 (Leubitz 2019/2021, Haverty 2021) опубликованы; **итогов фазы 2 в рецензируемой литературе нет** — только регистрации [NCT04135495](https://clinicaltrials.gov/study/NCT04135495), [NCT04126473](https://clinicaltrials.gov/study/NCT04126473) и пресс-релизы: ppFEV1 −0.31% к дню 35 (+2.83% у 6 «респондеров»), исходный хлорид пота предсказывал ответ (p=0.00013); программа терминирована в сентябре 2022 совместно с CFF — низкая лёгочная концентрация при п/к (~2 мкМ).

**Текущий статус (2026).** Переименован в **exaluren**; фокус Eloxx — нефрология: Альпорт (фаза 2a опубликована — Gale/JCI 2026: рост коллагена IV в биоптатах, первое прямое in vivo доказательство; идёт слепая фаза 2b EXACT [NCT07523581](https://clinicaltrials.gov/study/NCT07523581), чтение ~2027), ADPKD (ODD FDA/EC 2026, IND октябрь 2026, Phase 2 EXPRESS, topline 2029).

**Поправки наших прежних фактов (важно):**
- фазы 2 ELX-02 по DMD **не существовало** — DMD была только преклиникой (R3381X 35× in vitro, mdx-мыши); путаница с аталуреном (NCT03796637 — аталурен);
- слияния с Zevra не было — Eloxx независима (Nasdaq: ELOX);
- «уход в нефрологию» подтверждён, но с опубликованной фазой 2a и активными EXACT/EXPRESS.

## Значение для нашей работы

1. **ELX-02 — идеальный объект few-shot:** отсутствует в 8 препаратах Toledano → our профиль-калибровка (LODO 0.518) — рабочий путь предсказания; zero-shot по дескрипторам показал 0.142 (недостаточно). Для диссертации: ELX-02 = «невидимый препарат» с реальными внешними данными для проверки.
2. **Pranke 2023** — «ручная» версия нашей контекстной модели (UUGAC = +4C-контекст) — прямой предшественник, подкрепляет признак down_1nt.
3. **Destoop/Crawford** — готовые внешние тесты для ELX-02-специфичных предсказаний (206 + органоиды G542X).
4. **ML-ниша с ELX-02 пуста** — вычислительных работ нет вообще.

## Ограничения

Клинические итоги CF — только пресс-релизы (нельзя цитировать как рецензируемые); данные органоидов гетерогенны по протоколам; Gale 2026 — n=3 (доказательство концепции, не эффективность).

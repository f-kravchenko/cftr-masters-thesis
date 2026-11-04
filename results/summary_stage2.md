# Этап 2: тюнинг enet + drug-aware мультитаск + few-shot LODO

       split          model  spearman_mean  spearman_min  rmse_log_mean  n
cftr_holdout     enet_tuned          0.614         0.187          0.173  8
cftr_holdout         mt_hgb          0.621         0.290          0.165  8
  genegroup5     enet_tuned          0.707         0.287          0.151 40
  genegroup5         mt_hgb          0.713         0.312          0.145 40
lodo_fewshot mt_hgb_profile          0.654         0.319          0.352  8
     random5     enet_tuned          0.707         0.306          0.151 40
     random5         mt_hgb          0.713         0.287          0.145 40

Контекст: бустинг per-drug (этап 1) = 0.651/0.649/0.630 на random5/genegroup5/cftr_holdout; lodo ridge floor (этап 1) = 0.751, но он требует измерений 7 препаратов того же варианта. mt_hgb_profile использует только последовательность + 300 калибровочных измерений held-out препарата.

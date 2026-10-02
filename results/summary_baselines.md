# Бейзлайны и LOD-CV (промежуточный итог)

               split        model  spearman_mean  spearman_min  rmse_log_mean  n
        cftr_holdout         enet            NaN           NaN          0.253  8
        cftr_holdout          hgb          0.630         0.291          0.164  8
        cftr_holdout        rules          0.490         0.054          0.191  8
cftr_holdout_topdrug          hgb          0.592         0.592          0.803  1
          genegroup5         enet            NaN           NaN          0.244 45
          genegroup5          hgb          0.649         0.162          0.140 45
          genegroup5        rules          0.558         0.126          0.164 45
          lodo_ridge ridge_7drugs          0.751         0.387          0.147  8
             random5         enet            NaN           NaN          0.244 45
             random5          hgb          0.651         0.177          0.140 45
             random5        rules          0.560         0.143          0.164 45

Заметки: random5 — сравнимость со статьёй; genegroup5 — перенос на невидимые гены; cftr_holdout — train без CFTR, тест на 76 измеренных CFTR-вариантах; cftr_holdout_topdrug (spearman=top-1, rmse_log=top-2) — выбор лучшего препарата HGB; lodo_ridge — zero-shot floor для невидимого препарата (только межпрепаратная корреляция).

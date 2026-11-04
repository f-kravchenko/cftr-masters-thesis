# TransStop (pretrained) воспроизведение

Контекст-победитель (пилот): seq_context_12

       split                model       drug    n  r2_their  mae_their  spearman
  their_test transstop_pretrained        ALL 4839  0.939839   0.142882  0.965370
    cftr_all transstop_pretrained        ALL  692  0.961214   0.095048  0.973232
cftr_perdrug transstop_pretrained    CC90009   76       NaN        NaN  0.898674
cftr_perdrug transstop_pretrained  Clitocine   69       NaN        NaN  0.947022
cftr_perdrug transstop_pretrained        DAP   70       NaN        NaN  0.947406
cftr_perdrug transstop_pretrained        FUr   82       NaN        NaN  0.494977
cftr_perdrug transstop_pretrained       G418   77       NaN        NaN  0.941322
cftr_perdrug transstop_pretrained Gentamicin   77       NaN        NaN  0.481545
cftr_perdrug transstop_pretrained     SJ6986   79       NaN        NaN  0.969304
cftr_perdrug transstop_pretrained        SRI   79       NaN        NaN  0.966821
cftr_topdrug transstop_pretrained       top1   56       NaN        NaN  0.714286

Контекст: их test-сплит — модель видела CFTR в train (случайный сплит); cftr_* — оценка памяти модели, НЕ переноса. Наш cftr_holdout (etrain без CFTR): enet_tuned 0.614 / mt_hgb 0.621 / hgb 0.630 Spearman; top-1 0.592.

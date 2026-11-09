# Оценивание одноиндексных и многоиндексных моделей панельных данных с латентными факторами

Выпускная квалификационная работа (бакалавриат, факультет математики НИУ ВШЭ).

## Структура

- [`one-factor-model-visualization/`](one-factor-model-visualization/) — интерактивная визуализация одноиндексной модели
  $Y_{it} = c_i + \ell_i X_{it}^\top \beta + u_{it}$.
  Открыть в браузере: https://mark37mur-ops.github.io/diplom/one-factor-model-visualization/
- [`multi-factor-model-visualization/`](multi-factor-model-visualization/) — интерактивная визуализация многоиндексной модели
  $Y_{it} = c_i + X_{it}^\top(\ell_{i1}\beta_1 + \dots + \ell_{im}\beta_m) + u_{it}$.
  Открыть в браузере: https://mark37mur-ops.github.io/diplom/multi-factor-model-visualization/
- [`latent-factor-model-lab/`](latent-factor-model-lab/) — интерактивная лаборатория многоиндексной модели с латентными факторами
  $Y_{it} = c_i + X_{it}^\top P^\top \ell_i + \gamma_i^\top f_t + \varepsilon_{it}$:
  ползунки масштаба данных, сравнение с МНК и оракулом, эксперименты.
  Открыть в браузере: https://mark37mur-ops.github.io/diplom/latent-factor-model-lab/
- [`svd-pca-visualization/`](svd-pca-visualization/) — интерактивная визуализация сингулярного разложения и метода главных компонент
  $A = U\Sigma V^\top$.
  Открыть в браузере: https://mark37mur-ops.github.io/diplom/svd-pca-visualization/

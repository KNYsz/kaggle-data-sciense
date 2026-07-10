# 実験ログ

## 実験の流れ

1. もっとも単純なベースラインとして、`store_nbr` と `family` の平均売上を予測値にするモデルを作成し、Kaggle へ提出した。
2. holdout で CV を作成し、曜日と販促を使う階層平均へ切り替えた。
3. 実験ごとの CV と LB を `experiments/runs.csv` に保存し、`reports/cv_lb_scatter.png` を更新した。

## 結果

| run_name | model | CV RMSLE | LB RMSLE | 備考 |
| --- | --- | ---: | ---: | --- |
| baseline_mean | baseline | 0.69569 | 0.67811 | store-family 平均 |
| hier_mean | hierarchical_mean | 0.63335 | 0.60482 | store-family-dayofweek-promotion 階層平均 |

![CV/LB 散布図](../reports/cv_lb_scatter.png)

## 所感

- ベースラインでも Kaggle の提出形式は通ることを確認できた。
- holdout と LB の相関は完全ではないが、改善後のモデルは LB でも明確に改善した。
- 次にやるなら、店舗別・family 別の重み付け、曜日ごとの平滑化、外れ値処理を試す価値がある。

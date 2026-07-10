# データ仕様

## 主要ファイル

| ファイル | 役割 | 主な列 |
| --- | --- | --- |
| `train.csv` | 学習データ | `id`, `date`, `store_nbr`, `family`, `sales`, `onpromotion` |
| `test.csv` | 予測対象 | `id`, `date`, `store_nbr`, `family`, `onpromotion` |
| `stores.csv` | 店舗属性 | `store_nbr`, `city`, `state`, `type`, `cluster` |
| `oil.csv` | 原油価格 | `date`, `dcoilwtico` |
| `holidays_events.csv` | 祝日・イベント | `date`, `type`, `locale`, `locale_name`, `description`, `transferred` |
| `transactions.csv` | 店舗別取引量 | `date`, `store_nbr`, `transactions` |
| `sample_submission.csv` | 提出形式の確認用 | `id`, `sales` |

## 実データの確認結果

- `train.csv`: 3,000,888 行, 6 列
- `test.csv`: 28,512 行, 5 列
- 学習期間: 2013-01-01 から 2017-08-15
- 予測対象期間: 2017-08-16 から 2017-08-31
- 店舗数: 54
- 商品カテゴリ数: 33
- `oil.csv` の欠損: 43 件
- `sample_submission.csv` の行数: 28,512

## 提出形式

提出ファイルは `id`, `sales` の 2 列です。`sample_submission.csv` と行数を一致させる必要があります。

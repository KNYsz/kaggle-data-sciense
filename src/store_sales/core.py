from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"
EXPERIMENTS_DIR = ROOT / "experiments"

COMPETITION_SLUG = "store-sales-time-series-forecasting"
VALIDATION_START = pd.Timestamp("2017-08-01")

BASE_FEATURES = [
    "store_nbr",
    "onpromotion",
    "dcoilwtico",
    "is_holiday",
    "year",
    "month",
    "dayofmonth",
    "dayofweek",
    "weekofyear",
    "weekend",
    "city_enc",
    "state_enc",
    "type_enc",
    "cluster_enc",
    "family_enc",
    "store_family_mean",
    "family_mean",
    "store_mean",
    "dow_mean",
    "store_dow_mean",
]

LAG_FEATURES = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_roll_mean_7",
]

MODEL_FEATURES = BASE_FEATURES + LAG_FEATURES


@dataclass(frozen=True)
class DataBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    stores: pd.DataFrame
    oil: pd.DataFrame
    holidays: pd.DataFrame


def ensure_directories() -> None:
    for directory in [DATA_DIR, RAW_DIR, OUTPUT_DIR, REPORTS_DIR, EXPERIMENTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_raw_data(raw_dir: Path = RAW_DIR) -> DataBundle:
    train = pd.read_csv(raw_dir / "train.csv", parse_dates=["date"])
    test = pd.read_csv(raw_dir / "test.csv", parse_dates=["date"])
    stores = pd.read_csv(raw_dir / "stores.csv")
    oil = pd.read_csv(raw_dir / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(raw_dir / "holidays_events.csv", parse_dates=["date"])
    return DataBundle(train=train, test=test, stores=stores, oil=oil, holidays=holidays)


def build_holiday_frame(holidays: pd.DataFrame) -> pd.DataFrame:
    holiday_frame = holidays.loc[holidays["transferred"].eq(False), ["date"]].copy()
    holiday_frame["is_holiday"] = 1
    return holiday_frame.drop_duplicates(subset=["date"])


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["year"] = frame["date"].dt.year.astype("int16")
    frame["month"] = frame["date"].dt.month.astype("int8")
    frame["dayofmonth"] = frame["date"].dt.day.astype("int8")
    frame["dayofweek"] = frame["date"].dt.dayofweek.astype("int8")
    frame["weekofyear"] = frame["date"].dt.isocalendar().week.astype("int16")
    frame["weekend"] = frame["dayofweek"].isin([5, 6]).astype("int8")
    return frame


def encode_categories(frame: pd.DataFrame, mappings: dict[str, dict[str, int]] | None = None) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    frame = frame.copy()
    mappings = {} if mappings is None else {key: value.copy() for key, value in mappings.items()}
    for column in ["city", "state", "type", "cluster", "family"]:
        if column not in frame.columns:
            continue
        if column not in mappings:
            values = pd.Index(frame[column].astype(str).fillna("__missing__").unique()).sort_values()
            mappings[column] = {value: index for index, value in enumerate(values)}
        encoded = frame[column].astype(str).fillna("__missing__").map(mappings[column]).fillna(-1)
        frame[f"{column}_enc"] = encoded.astype("int32")
    return frame, mappings


def merge_auxiliary_data(frame: pd.DataFrame, bundle: DataBundle) -> pd.DataFrame:
    holiday_frame = build_holiday_frame(bundle.holidays)

    oil = bundle.oil.sort_values("date").copy()
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()

    merged = frame.merge(bundle.stores, on="store_nbr", how="left")
    merged = merged.merge(oil[["date", "dcoilwtico"]], on="date", how="left")
    merged = merged.merge(holiday_frame, on="date", how="left")
    merged["is_holiday"] = merged["is_holiday"].fillna(0).astype("int8")
    merged["dcoilwtico"] = merged["dcoilwtico"].ffill().bfill()
    merged = add_calendar_features(merged)
    return merged


def build_target_encodings(frame: pd.DataFrame, target: str = "sales") -> dict[str, pd.Series]:
    global_mean = frame[target].mean()
    encodings = {
        "store_family_mean": frame.groupby(["store_nbr", "family"])[target].mean(),
        "family_mean": frame.groupby("family")[target].mean(),
        "store_mean": frame.groupby("store_nbr")[target].mean(),
        "dow_mean": frame.groupby("dayofweek")[target].mean(),
        "store_dow_mean": frame.groupby(["store_nbr", "dayofweek"])[target].mean(),
        "global_mean": pd.Series(global_mean),
    }
    return encodings


def apply_target_encodings(frame: pd.DataFrame, encodings: dict[str, pd.Series]) -> pd.DataFrame:
    frame = frame.copy()
    global_mean = float(encodings["global_mean"].iloc[0])

    frame = frame.merge(
        encodings["store_family_mean"].reset_index(name="store_family_mean"),
        on=["store_nbr", "family"],
        how="left",
    )
    frame = frame.merge(encodings["family_mean"].reset_index(name="family_mean"), on=["family"], how="left")
    frame = frame.merge(encodings["store_mean"].reset_index(name="store_mean"), on=["store_nbr"], how="left")
    frame = frame.merge(encodings["dow_mean"].reset_index(name="dow_mean"), on=["dayofweek"], how="left")
    frame = frame.merge(
        encodings["store_dow_mean"].reset_index(name="store_dow_mean"),
        on=["store_nbr", "dayofweek"],
        how="left",
    )

    for column in ["store_family_mean", "family_mean", "store_mean", "dow_mean", "store_dow_mean"]:
        frame[column] = frame[column].fillna(global_mean).astype("float32")

    return frame


def add_time_features(frame: pd.DataFrame, bundle: DataBundle) -> pd.DataFrame:
    merged = merge_auxiliary_data(frame, bundle)
    merged, _ = encode_categories(merged)
    return merged


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["store_nbr", "family", "date"]).copy()
    grouped_sales = frame.groupby(["store_nbr", "family"], sort=False)["sales"]
    frame["sales_lag_1"] = grouped_sales.shift(1)
    frame["sales_lag_7"] = grouped_sales.shift(7)
    frame["sales_lag_14"] = grouped_sales.shift(14)
    frame["sales_roll_mean_7"] = grouped_sales.transform(lambda series: series.shift(1).rolling(7).mean())
    return frame


def rmsle(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return float(np.sqrt(mean_squared_log_error(np.clip(y_true, 0, None), np.clip(y_pred, 0, None))))


def baseline_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    global_mean = float(train["sales"].mean())
    store_family_mean = train.groupby(["store_nbr", "family"])["sales"].mean().reset_index(name="store_family_mean")
    family_mean = train.groupby("family")["sales"].mean().reset_index(name="family_mean")
    store_mean = train.groupby("store_nbr")["sales"].mean().reset_index(name="store_mean")

    prediction = test[["id", "store_nbr", "family"]].copy()
    prediction = prediction.merge(store_family_mean, on=["store_nbr", "family"], how="left")
    prediction = prediction.merge(family_mean, on=["family"], how="left")
    prediction = prediction.merge(store_mean, on=["store_nbr"], how="left")
    prediction["sales"] = (
        prediction["store_family_mean"]
        .fillna(prediction["family_mean"])
        .fillna(prediction["store_mean"])
        .fillna(global_mean)
        .clip(lower=0)
    )
    return prediction[["id", "sales"]]


def make_holdout_split(frame: pd.DataFrame, validation_start: pd.Timestamp = VALIDATION_START) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_part = frame.loc[frame["date"] < validation_start].copy()
    valid_part = frame.loc[frame["date"] >= validation_start].copy()
    return train_part, valid_part


def train_xgb_model(train_frame: pd.DataFrame, valid_frame: pd.DataFrame | None = None, seed: int = 42) -> XGBRegressor:
    train_features, encoders = encode_categories(train_frame)
    train_features = apply_target_encodings(train_features, build_target_encodings(train_features))
    train_features = add_lag_features(train_features)
    train_features = train_features.loc[train_features["date"] >= pd.Timestamp("2016-01-01")].dropna(subset=LAG_FEATURES)

    x_train = train_features[MODEL_FEATURES]
    y_train = np.log1p(train_features["sales"].astype(float))

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        tree_method="hist",
        objective="reg:squarederror",
    )
    model.fit(x_train, y_train, verbose=False)
    return model


def fit_predict_xgb(train_frame: pd.DataFrame, valid_frame: pd.DataFrame, test_frame: pd.DataFrame, seed: int = 42) -> tuple[XGBRegressor, np.ndarray, np.ndarray]:
    train_features, encoders = encode_categories(train_frame)
    encodings = build_target_encodings(train_features)

    model_train = apply_target_encodings(train_features.copy(), encodings)
    model_train = add_lag_features(model_train)
    model_train = model_train.loc[model_train["date"] >= pd.Timestamp("2016-01-01")].dropna(subset=LAG_FEATURES)

    x_train = model_train[MODEL_FEATURES]
    y_train = np.log1p(model_train["sales"].astype(float))

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        tree_method="hist",
        objective="reg:squarederror",
    )
    model.fit(x_train, y_train, verbose=False)

    valid_features, _ = encode_categories(valid_frame, encoders)
    valid_features = apply_target_encodings(valid_features, encodings)
    valid_features = recursive_lag_forecast(model, model_train, valid_features, seed=seed)

    test_features, _ = encode_categories(test_frame, encoders)
    test_features = apply_target_encodings(test_features, encodings)
    test_features = recursive_lag_forecast(model, model_train, test_features, seed=seed)

    valid_pred = np.clip(np.expm1(model.predict(valid_features[MODEL_FEATURES])), 0, None)
    test_pred = np.clip(np.expm1(model.predict(test_features[MODEL_FEATURES])), 0, None)
    return model, valid_pred, test_pred


def recursive_lag_forecast(
    model: XGBRegressor,
    history_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    history = {
        (store_nbr, family): group["sales"].astype(float).tolist()
        for (store_nbr, family), group in history_frame.sort_values(["date", "store_nbr", "family"]).groupby(["store_nbr", "family"], sort=False)
    }

    future = future_frame.sort_values(["date", "store_nbr", "family"]).copy()
    future["sales_lag_1"] = np.nan
    future["sales_lag_7"] = np.nan
    future["sales_lag_14"] = np.nan
    future["sales_roll_mean_7"] = np.nan

    predicted_values: list[float] = []
    for date_value in future["date"].drop_duplicates().sort_values():
        date_mask = future["date"].eq(date_value)
        date_rows = future.loc[date_mask].copy()

        lag_1_values = []
        lag_7_values = []
        lag_14_values = []
        roll_mean_values = []

        for _, row in date_rows.iterrows():
            key = (row["store_nbr"], row["family"])
            series = history.get(key, [])
            lag_1 = series[-1] if len(series) >= 1 else 0.0
            lag_7 = series[-7] if len(series) >= 7 else lag_1
            lag_14 = series[-14] if len(series) >= 14 else lag_7
            roll_mean = float(np.mean(series[-7:])) if len(series) >= 7 else float(np.mean(series)) if series else 0.0

            lag_1_values.append(lag_1)
            lag_7_values.append(lag_7)
            lag_14_values.append(lag_14)
            roll_mean_values.append(roll_mean)

        future.loc[date_mask, "sales_lag_1"] = lag_1_values
        future.loc[date_mask, "sales_lag_7"] = lag_7_values
        future.loc[date_mask, "sales_lag_14"] = lag_14_values
        future.loc[date_mask, "sales_roll_mean_7"] = roll_mean_values

        predictions = np.clip(np.expm1(model.predict(future.loc[date_mask, MODEL_FEATURES])), 0, None)
        predicted_values.extend(predictions.tolist())

        for (_, row), prediction in zip(date_rows.iterrows(), predictions, strict=False):
            key = (row["store_nbr"], row["family"])
            history.setdefault(key, []).append(float(prediction))

    future = future.sort_values(["date", "store_nbr", "family"]).copy()
    future["forecast_sales"] = predicted_values
    return future


def save_submission(prediction: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(path, index=False)
    return path


def hierarchical_mean_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()
    test = test.copy()
    train["dayofweek"] = train["date"].dt.dayofweek
    test["dayofweek"] = test["date"].dt.dayofweek

    global_mean = float(train["sales"].mean())
    mean_sfdp = train.groupby(["store_nbr", "family", "dayofweek", "onpromotion"])["sales"].mean()
    mean_sfd = train.groupby(["store_nbr", "family", "dayofweek"])["sales"].mean()
    mean_sf = train.groupby(["store_nbr", "family"])["sales"].mean()
    mean_fd = train.groupby(["family", "dayofweek"])["sales"].mean()
    mean_d = train.groupby("dayofweek")["sales"].mean()

    predictions = []
    for row in test.itertuples(index=False):
        key_sfdp = (row.store_nbr, row.family, row.dayofweek, row.onpromotion)
        key_sfd = (row.store_nbr, row.family, row.dayofweek)
        key_sf = (row.store_nbr, row.family)
        key_fd = (row.family, row.dayofweek)
        prediction = mean_sfdp.get(
            key_sfdp,
            mean_sfd.get(
                key_sfd,
                mean_sf.get(key_sf, mean_fd.get(key_fd, mean_d.get(row.dayofweek, global_mean))),
            ),
        )
        predictions.append(max(float(prediction), 0.0))

    return pd.DataFrame({"id": test["id"], "sales": predictions})

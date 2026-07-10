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
        frame[f"{column}_enc"] = frame[column].astype(str).fillna("__missing__").map(mappings[column]).astype("int32")
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

    x_train = train_features[BASE_FEATURES]
    y_train = np.log1p(train_features["sales"].astype(float))

    eval_set = None
    if valid_frame is not None:
        valid_features, _ = encode_categories(valid_frame, encoders)
        valid_features = apply_target_encodings(valid_features, build_target_encodings(train_features))
        eval_set = [(valid_features[BASE_FEATURES], np.log1p(valid_features["sales"].astype(float)))]

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        tree_method="hist",
        objective="reg:squarederror",
    )
    model.fit(x_train, y_train, eval_set=eval_set, verbose=False)
    return model


def fit_predict_xgb(train_frame: pd.DataFrame, valid_frame: pd.DataFrame, test_frame: pd.DataFrame, seed: int = 42) -> tuple[XGBRegressor, np.ndarray, np.ndarray]:
    model = train_xgb_model(train_frame, valid_frame=valid_frame, seed=seed)

    train_features, encoders = encode_categories(train_frame)
    encodings = build_target_encodings(train_features)
    valid_features, _ = encode_categories(valid_frame, encoders)
    valid_features = apply_target_encodings(valid_features, encodings)
    test_features, _ = encode_categories(test_frame, encoders)
    test_features = apply_target_encodings(test_features, encodings)

    valid_pred = np.clip(np.expm1(model.predict(valid_features[BASE_FEATURES])), 0, None)
    test_pred = np.clip(np.expm1(model.predict(test_features[BASE_FEATURES])), 0, None)
    return model, valid_pred, test_pred


def save_submission(prediction: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(path, index=False)
    return path

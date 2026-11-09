"""Chronological, purged, held-out evaluation; benchmark against no-change price."""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from .features import ALL_FEATURES, PRICE_FEATURES, eligible_training_rows


def _make_model():
    return make_pipeline(SimpleImputer(strategy="constant", fill_value=0, add_indicator=False),
                         HistGradientBoostingRegressor(max_iter=130, max_leaf_nodes=7, l2_regularization=5, learning_rate=0.04, min_samples_leaf=10, random_state=42))


def metrics(rows: pd.DataFrame, predicted_returns: np.ndarray) -> dict:
    target = rows.target_return.to_numpy(dtype=float)
    last_close = rows.close.to_numpy(dtype=float)
    predicted_price = last_close * (1 + predicted_returns)
    true_price = rows.target_close.to_numpy(dtype=float)
    return {"n": len(rows), "mae_price": float(mean_absolute_error(true_price, predicted_price)),
            "rmse_price": float(np.sqrt(mean_squared_error(true_price, predicted_price))),
            "directional_accuracy": float(np.mean(np.sign(target) == np.sign(predicted_returns))),
            "return_mae": float(mean_absolute_error(target, predicted_returns))}


def train_evaluate(frame: pd.DataFrame, *, min_rows: int = 60, holdout_fraction: float = 0.2, cv_splits: int = 3) -> tuple[dict, dict]:
    data = eligible_training_rows(frame, min_rows=min_rows)
    n = len(data)
    test_n = max(10, int(np.ceil(n*holdout_fraction)))
    # Exclude one extra supervised row: its T+1 outcome would be the first holdout date.
    train = data.iloc[:n-test_n-1]
    test = data.iloc[n-test_n:]
    if len(train) < 35:
        raise ValueError("Not enough training sessions after chronological held-out split")
    result = {"rows_total":n, "train_rows":len(train), "test_rows":len(test),
              "train_last_session":str(train.iloc[-1].session), "test_first_session":str(test.iloc[0].session),
              "purge_sessions":1, "models":{}}
    fitted = {}
    for key, cols in (("price_only",PRICE_FEATURES),("price_plus_sentiment",ALL_FEATURES)):
        model = _make_model()
        # Validation is entirely within pre-test history, with one-row purging.
        cv_results = []
        if len(train) >= 55:
            splitter = TimeSeriesSplit(n_splits=cv_splits, gap=1)
            for i,j in splitter.split(train):
                if len(i) < 15 or len(j) < 5:
                    continue
                trial = _make_model()
                trial.fit(train.iloc[i][cols], train.iloc[i].target_return)
                cv_results.append(metrics(train.iloc[j], trial.predict(train.iloc[j][cols])))
        model.fit(train[cols], train.target_return)
        predicted = model.predict(test[cols])
        result["models"][key] = {"test":metrics(test,predicted), "validation_folds":cv_results,
                                  "predicted_prices":(test.close*(1+predicted)).round(5).tolist()}
        fitted[key] = {"model": model, "features": cols}
    result["models"]["no_change_baseline"] = {"test": metrics(test, np.zeros(len(test)))}
    result["test_sessions"] = test.session.astype(str).tolist()
    result["actual_prices"] = test.target_close.astype(float).round(5).tolist()
    return result, fitted


def save_artifacts(metrics_dict: dict, models: dict, folder: str | Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"evaluation.json").write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")
    joblib.dump(models, folder/"models.joblib")  # Load only your own trusted model artifacts.


def predict_next(last_row: pd.Series, trained: dict, *, threshold: float = 0.005) -> dict:
    if int(last_row.source_count) < 2:
        raise ValueError("Latest session lacks two observed social sources; cannot claim a complete sentiment forecast")
    cols = trained["features"]
    values = pd.DataFrame([{col: last_row[col] for col in cols}])
    estimated_return = float(trained["model"].predict(values)[0])
    direction = "up" if estimated_return > 0 else ("down" if estimated_return < 0 else "flat")
    position = "long_candidate" if estimated_return >= threshold else ("short_candidate" if estimated_return <= -threshold else "no_position")
    return {"ticker":str(last_row.ticker), "as_of_session":str(last_row.session),
            "current_close":float(last_row.close), "predicted_next_close":float(last_row.close*(1+estimated_return)),
            "predicted_return":estimated_return, "direction":direction, "position":position,
            "social_mentions":int(last_row.total_social_mentions),
            "social_sentiment":float(last_row.social_sentiment),
            "news_sentiment":None if pd.isna(last_row.news_sentiment) else float(last_row.news_sentiment),
            "support":None if pd.isna(last_row.support) else float(last_row.support),
            "resistance":None if pd.isna(last_row.resistance) else float(last_row.resistance),
            "disclaimer":"Research-only prediction; not an executable trading instruction."}

"""Dashboard data, with strict separation between illustrative demo and local results.

The public API never starts third-party scraping, executes notebooks, or trains ML models.
Demo forecasts and evaluation numbers are SYNTHETIC; real outputs are strictly read
from existing artifacts produced by the research pipeline.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from news_predictor.features import build_features
from news_predictor.report import summarize_ticker

Mode = Literal["demo", "real"]
SortBy = Literal["mentions", "sentiment", "direction"]

# Deliberately fixed synthetic dataset date: the deployed UI is NOT a live market feed.
DEMO_AS_OF = "2026-11-13"
DEMO_BASE = {
    "0700.HK": (498.40, 475, 0.56, 0.21, 0.0094),
    "9988.HK": (151.30, 386, 0.31, 0.14, 0.0043),
    "1810.HK": (43.75, 315, -0.27, -0.13, -0.0066),
    "3690.HK": (101.80, 251, -0.46, -0.30, -0.0081),
    "1211.HK": (114.25, 293, 0.39, 0.19, 0.0070),
}


def repo_root() -> Path:
    env = os.getenv("STOCK_PREDICTOR_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # .../repo/src/news_predictor/web/data.py -> repo
    return Path(__file__).resolve().parents[3]


def assets() -> dict[str, dict]:
    config = repo_root() / "config" / "assets.json"
    return {a["ticker"]: a for a in json.loads(config.read_text(encoding="utf-8"))}


def clean_number(value: Any, decimals: int | None = 3) -> float | None:
    try:
        v = float(value)
        if not math.isfinite(v):
            return None
        return round(v, decimals) if decimals is not None else v
    except (ValueError, TypeError):
        return None


def _next_business_day(day: str) -> str:
    # Demo-only convenience; this is not a Hong Kong exchange holiday calendar.
    return (pd.Timestamp(day) + pd.offsets.BDay(1)).date().isoformat()


def _synthetic_asset(asset: dict) -> dict:
    ticker = asset["ticker"]
    idx = list(assets()).index(ticker)
    base, mentions, sentiment, news_sentiment, estimate = DEMO_BASE[ticker]
    rng = np.random.default_rng(8625 + idx * 179)
    dates = pd.bdate_range(end=DEMO_AS_OF, periods=70)
    returns = rng.normal(loc=estimate / 7, scale=0.011 + idx * .0012, size=len(dates))
    closes = base * np.exp(np.cumsum(returns) - np.sum(returns))
    closes[-1] = base
    history = []
    for i, d in enumerate(dates):
        daily_sentiment = float(np.clip(sentiment + 0.19 * np.sin(i / 5.5) + rng.normal(0, .12), -1, 1))
        history.append({"session": d.date().isoformat(), "close": round(float(closes[i]), 2),
                        "sentiment": round(daily_sentiment, 3),
                        "mentions": max(0, int(mentions / 7 + 13 * np.sin(i / 6) + rng.normal(0, 8)))})
    last20 = closes[-20:]
    support = float(np.min(last20) * 0.985)
    resistance = float(np.max(last20) * 1.015)
    test_rows = history[-18:]
    actual = np.array([r["close"] for r in test_rows], dtype=float)
    predicted = actual + rng.normal(0, base * .007, len(actual))
    baseline = np.r_[closes[-19], actual[:-1]]
    mae = float(np.mean(np.abs(predicted - actual)))
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    positions = "long_candidate" if estimate >= .005 else "short_candidate" if estimate <= -.005 else "no_position"
    source_split = {"lihkg": round(mentions * .61), "futu": mentions - round(mentions * .61)}
    return {
        "ticker": ticker, "asset_name": asset["name"], "exchange": asset["exchange"],
        "as_of_session": DEMO_AS_OF, "currency": "HKD", "current_close": round(base, 2),
        "predicted_next_close": round(base * (1 + estimate), 2),
        "predicted_return": round(estimate, 5),
        "forecast_session": _next_business_day(DEMO_AS_OF), "direction": "up" if estimate > 0 else "down" if estimate < 0 else "flat",
        "position": positions, "support": round(support, 2), "resistance": round(resistance, 2),
        "social_mentions": mentions, "social_sentiment": sentiment, "news_sentiment": news_sentiment,
        "source_count": 2, "source_mentions": source_split, "news_mentions": 22 + idx * 3,
        "history": history, "evaluation": {
            "kind": "synthetic_illustration", "sample_count": len(test_rows),
            "price_mae": round(mae, 3), "price_rmse": round(rmse, 3),
            "baseline_mae": round(float(np.mean(np.abs(baseline - actual))), 3),
            "test_series": [
                {"session": row["session"], "actual": row["close"], "predicted": round(float(pred), 2)}
                for row, pred in zip(test_rows, predicted)
            ],
        },
        "prediction_status": "Illustrative synthetic prediction — no trained model",
        "data_status": "synthetic_demo", "is_demo": True,
    }


@lru_cache(maxsize=1)
def _demo_all() -> dict[str, dict]:
    return {ticker: _synthetic_asset(a) for ticker, a in assets().items()}


def real_asset(asset: dict) -> dict | None:
    """Load saved results only; never scrape or train in a public HTTP request."""
    ticker = asset["ticker"]
    folder = repo_root() / "data" / "raw" / ticker
    if not all((folder / name).is_file() for name in ("prices.csv", "social.csv", "news.csv")):
        return None
    try:
        prices, social, news = [pd.read_csv(folder / name) for name in ("prices.csv", "social.csv", "news.csv")]
        # Sentiment must already have been computed by the notebook / CLI.
        if prices.empty or (not social.empty and "sentiment" not in social.columns) or (not news.empty and "sentiment" not in news.columns):
            return None
        for f in (social, news):
            if "sentiment" not in f:
                f["sentiment"] = pd.Series(dtype=float)
        frame = build_features(prices, social, news)
        if frame.empty:
            return None
        last = frame.iloc[-1]
        summary = summarize_ticker(asset, frame, trained=None)
        # Unsafe pickle/joblib loading is deliberately forbidden in public HTTP handlers.
        # A validated saved evaluation is displayed separately, not presented as a live forecast.
        evaluation_path = repo_root() / "outputs" / ticker / "evaluation.json"
        evaluation = None
        if evaluation_path.is_file():
            result = json.loads(evaluation_path.read_text(encoding="utf-8"))
            model = result.get("models", {}).get("price_plus_sentiment", {})
            baseline = result.get("models", {}).get("no_change_baseline", {})
            test = model.get("test", {})
            if test and len(result.get("test_sessions", [])) == len(model.get("predicted_prices", [])) == len(result.get("actual_prices", [])):
                evaluation = {
                    "kind": "historical_out_of_sample", "sample_count": test.get("n"),
                    "price_mae": clean_number(test.get("mae_price")),
                    "price_rmse": clean_number(test.get("rmse_price")),
                    "directional_accuracy": clean_number(test.get("directional_accuracy")),
                    "baseline_mae": clean_number(baseline.get("test", {}).get("mae_price")),
                    "test_series": [
                        {"session": str(d), "actual": clean_number(y), "predicted": clean_number(p)}
                        for d, y, p in zip(result["test_sessions"], result["actual_prices"], model["predicted_prices"])
                    ],
                }
        history = [{"session": str(row.session), "close": clean_number(row.close, 2),
                    "sentiment": clean_number(row.social_sentiment),
                    "mentions": int(row.total_social_mentions)}
                   for row in frame.tail(70).itertuples(index=False)]
        # File dates and completeness are explicitly visible. Latest records may be stale.
        return {
            **summary, "currency": "HKD", "forecast_session": None,
            "predicted_return": None, "source_count": int(last.source_count),
            "source_mentions": {"lihkg": int(last.lihkg_mentions), "futu": int(last.futu_mentions)},
            "news_mentions": int(last.news_mentions), "history": history, "evaluation": evaluation,
            "data_status": "local_real_data", "is_demo": False,
            "prediction_status": "Latest forecast unavailable via read-only API; historical evaluation shown if saved",
        }
    except (ValueError, KeyError, TypeError, pd.errors.ParserError, OSError):
        # Never leak local directories, credential strings or stack traces in public responses.
        return None


def get_asset(ticker: str, mode: Mode) -> dict | None:
    a = assets().get(ticker)
    if a is None:
        return None
    if mode == "demo":
        return _demo_all()[ticker]
    return real_asset(a)


def overview(mode: Mode = "demo", sort_by: SortBy = "mentions") -> dict:
    all_assets = assets()
    items = [get_asset(t, mode) for t in all_assets]
    available = [a for a in items if a is not None]
    key = {"mentions": lambda a: a["social_mentions"],
           "sentiment": lambda a: a.get("social_sentiment") if a.get("social_sentiment") is not None else -2,
           "direction": lambda a: a.get("predicted_return") if a.get("predicted_return") is not None else -2}[sort_by]
    available.sort(key=key, reverse=True)
    rows = [{k: v for k, v in item.items() if k not in ("history", "evaluation")} for item in available]
    return {
        "mode": mode, "is_demo": mode == "demo", "as_of": DEMO_AS_OF if mode == "demo" else max((a["as_of_session"] for a in available), default=None),
        "notice": ("SYNTHETIC DEMONSTRATION — all prices, counts, sentiment, forecasts and backtests are invented for interface testing."
                   if mode == "demo" else "LOCAL RESEARCH DATA — not live quotes; forecasts unavailable until explicitly exported from a verified model."),
        "assets_configured": len(all_assets), "assets_available": len(available),
        "total_social_mentions": sum(a["social_mentions"] for a in available),
        "positive_sentiment_assets": sum(1 for a in available if a["social_sentiment"] is not None and a["social_sentiment"] > 0),
        "upward_signals": sum(1 for a in available if a["direction"] == "up"),
        "assets": rows,
    }

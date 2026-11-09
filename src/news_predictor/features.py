"""UTC-to-HK-close bucket alignment and leakage-aware daily features."""
from __future__ import annotations
import numpy as np
import pandas as pd

SENTIMENT_FEATURES = ["lihkg_mentions", "lihkg_sentiment", "futu_mentions", "futu_sentiment", "news_mentions", "news_sentiment", "total_social_mentions", "social_sentiment", "social_engagement", "source_count"]
PRICE_FEATURES = ["return_1d", "return_3d", "volatility_5", "volume_change", "price_to_ma5", "support_distance", "resistance_distance"]
ALL_FEATURES = PRICE_FEATURES + SENTIMENT_FEATURES


def market_cutoff_utc(sessions: pd.Series) -> pd.Series:
    """HKEX regular close 16:00 HKT; half-days / disruptions not calendar-corrected.

    For serious research substitute an actual exchange calendar before claiming results.
    """
    day = pd.to_datetime(sessions.astype(str))
    return day.dt.tz_localize("Asia/Hong_Kong") .add(pd.Timedelta(hours=16)).dt.tz_convert("UTC")


def _aggregate_posts(frame: pd.DataFrame, cutoffs: pd.Series, prefix: str, source: str | None = None) -> pd.DataFrame:
    output = pd.DataFrame({"session_idx": np.arange(len(cutoffs)), f"{prefix}_mentions": 0, f"{prefix}_sentiment": np.nan})
    if frame.empty:
        return output
    subset = frame.copy()
    if source is not None:
        subset = subset[subset["source"] == source]
    if subset.empty:
        return output
    subset["published_at"] = pd.to_datetime(subset["published_at"], utc=True, errors="coerce")
    subset = subset.dropna(subset=["published_at", "sentiment"])
    if subset.empty:
        return output
    # Searchsorted: publication after today's close goes to NEXT available trading session.
    indices = cutoffs.searchsorted(subset["published_at"], side="left")
    subset = subset.assign(session_idx=indices)
    subset = subset[subset.session_idx < len(cutoffs)]
    if subset.empty:
        return output
    agg = subset.groupby("session_idx")["sentiment"].agg(["count", "mean"])
    output.loc[agg.index, f"{prefix}_mentions"] = agg["count"].to_numpy()
    output.loc[agg.index, f"{prefix}_sentiment"] = agg["mean"].to_numpy()
    return output


def build_features(prices: pd.DataFrame, social: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "session", "close", "high", "low", "volume"}
    if not required.issubset(prices.columns):
        raise ValueError(f"Price dataframe missing {required - set(prices.columns)}")
    if prices["ticker"].nunique() != 1:
        raise ValueError("Build features separately per ticker")
    df = prices.sort_values("session").drop_duplicates("session").copy().reset_index(drop=True)
    df["cutoff_utc"] = market_cutoff_utc(df["session"])
    if not df["cutoff_utc"].is_monotonic_increasing:
        raise ValueError("Price sessions not chronologically sorted")
    df["return_1d"] = df.close.pct_change(fill_method=None)
    df["return_3d"] = df.close.pct_change(3, fill_method=None)
    df["volatility_5"] = df.return_1d.rolling(5).std()
    df["volume_change"] = df.volume.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    df["price_to_ma5"] = df.close / df.close.rolling(5).mean() - 1
    df["support"] = df.low.rolling(20, min_periods=5).min()
    df["resistance"] = df.high.rolling(20, min_periods=5).max()
    df["support_distance"] = df.close / df.support - 1
    df["resistance_distance"] = df.resistance / df.close - 1
    for source in ("lihkg", "futu"):
        part = _aggregate_posts(social, df.cutoff_utc, source, source)
        df[f"{source}_mentions"] = part[f"{source}_mentions"].to_numpy()
        df[f"{source}_sentiment"] = part[f"{source}_sentiment"].to_numpy()
    part = _aggregate_posts(news, df.cutoff_utc, "news")
    df["news_mentions"] = part["news_mentions"].to_numpy()
    df["news_sentiment"] = part["news_sentiment"].to_numpy()
    df["total_social_mentions"] = df.lihkg_mentions + df.futu_mentions
    numer = df.lihkg_sentiment.fillna(0)*df.lihkg_mentions + df.futu_sentiment.fillna(0)*df.futu_mentions
    df["social_sentiment"] = (numer / df.total_social_mentions.replace(0, np.nan))
    df["source_count"] = (df.lihkg_mentions>0).astype(int)+(df.futu_mentions>0).astype(int)
    if social.empty or "likes" not in social.columns:
        df["social_engagement"] = np.nan
    else:
        engagements = social.copy()
        engagements["sentiment"] = pd.to_numeric(engagements["likes"], errors="coerce").fillna(0).clip(lower=0)
        # The engagement statistic is mean likes, not the sentiment.
        en = _aggregate_posts(engagements, df.cutoff_utc, "likes")
        df["social_engagement"] = en["likes_sentiment"].to_numpy()
    df["target_return"] = df.close.shift(-1) / df.close - 1
    df["target_close"] = df.close.shift(-1)
    return df


def eligible_training_rows(frame: pd.DataFrame, *, min_rows: int = 60) -> pd.DataFrame:
    """Only train on observed two-source sentiment and known subsequent outcomes."""
    required = ALL_FEATURES + ["target_return", "target_close", "close", "source_count"]
    if any(k not in frame for k in required):
        raise ValueError("Run build_features first")
    usable = frame.loc[(frame.source_count == 2) & frame.target_return.notna() & frame[PRICE_FEATURES].notna().all(axis=1)].copy()
    usable = usable.replace([np.inf, -np.inf], np.nan)
    usable = usable.dropna(subset=PRICE_FEATURES+["target_return"])
    if len(usable) < min_rows:
        raise ValueError(f"Only {len(usable)} fully observed trading sessions; need at least {min_rows} for a defensible exploratory backtest. Do not treat the ten-day scrape as sufficient.")
    return usable.reset_index(drop=True)

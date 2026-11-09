"""Documented providers, provenance-preserving normalization, and no silent fallbacks.

LIHKG Actor: https://apify.com/claude_code_reviewer/lihkg-scraper-en
Apify client: https://docs.apify.com/api/client/python
Futu CLI: https://open.futunn.com/cli/research (requires authorization)
Exa SDK: https://github.com/exa-labs/exa-py
Yahoo research prototype: https://ranaroussi.github.io/yfinance/
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


LIHKG_ACTOR_ID = "claude_code_reviewer/lihkg-scraper-en"
SOCIAL_COLS = ["ticker", "source", "item_id", "url", "published_at", "text", "likes", "reposts", "views", "author_followers", "retrieved_at"]
NEWS_COLS = ["ticker", "source", "item_id", "url", "published_at", "text", "retrieved_at"]


def _timestamp(value: Any) -> pd.Timestamp | None:
    """Never replace missing publication time with scrape time."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        unit = "ms" if abs(value) > 1e11 else "s"
        stamp = pd.to_datetime(value, unit=unit, utc=True, errors="coerce")
    else:
        stamp = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(stamp) else stamp


def _records(payload: Any) -> list[dict]:
    """Accept list or documented-style result wrappers, reject unrecognized schemas."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("items", "results", "posts", "data", "list", "rows"):
            nested = payload.get(key)
            if isinstance(nested, list):
                return _records(nested)
            if isinstance(nested, dict):
                try:
                    return _records(nested)
                except ValueError:
                    pass
    raise ValueError("Source response has no recognized list of posts; inspect source JSON first")


def normalize_social(payload: Any, ticker: str, source: str, *, since: datetime | None = None, until: datetime | None = None) -> pd.DataFrame:
    """Only rows with genuine text and published timestamps survive. No missing metrics invented."""
    retrieved = pd.Timestamp.now(tz="UTC")
    output = []
    for row in _records(payload):
        created = _timestamp(row.get("created_at") or row.get("published_at") or row.get("timestamp") or row.get("time") or row.get("createdAt"))
        if created is None or (since and created < _timestamp(since)) or (until and created > _timestamp(until)):
            continue
        own = row.get("own_text")
        body = own if own is not None else (row.get("text") or row.get("content") or row.get("body") or row.get("message"))
        title = row.get("title") or ""
        body = str(body or "").strip()
        text = (str(title).strip() + " " + body).strip()
        if not text:
            continue
        author = row.get("author")
        followers = row.get("author_followers") or row.get("followers_count")
        if followers is None and isinstance(author, dict):
            followers = author.get("followers_count")
        item_id = row.get("reply_id") or row.get("post_id") or row.get("id") or row.get("thread_id") or row.get("url") or row.get("thread_url")
        if item_id is None:
            # Require stable identity for safe deduplication.
            continue
        output.append({
            "ticker": ticker, "source": source, "item_id": str(item_id),
            "url": row.get("thread_url") or row.get("post_url") or row.get("url"),
            "published_at": created, "text": text,
            "likes": row.get("like_count", row.get("likes")), "reposts": row.get("reposts"),
            "views": row.get("views"), "author_followers": followers,
            "retrieved_at": retrieved,
        })
    if not output:
        return pd.DataFrame(columns=SOCIAL_COLS)
    return pd.DataFrame(output, columns=SOCIAL_COLS).drop_duplicates(["ticker", "source", "item_id"])


def run_lihkg_actor(asset: dict, *, lookback_days: int = 10, limit_per_alias: int = 100, token: str | None = None) -> tuple[pd.DataFrame, dict]:
    from apify_client import ApifyClient
    token = token or os.getenv("APIFY_TOKEN")
    if not token:
        raise RuntimeError("Set APIFY_TOKEN in .env before running a paid Actor")
    now = datetime.now(timezone.utc)
    # Actor's documented dateFrom accepts relative '10d'. Order by original creation time.
    request = {"mode": "search", "keywords": asset["aliases"], "maxItems": limit_per_alias,
               "order": "desc_create_time", "dateFrom": f"{lookback_days}d", "aiLabels": False}
    client = ApifyClient(token)
    run = client.actor(LIHKG_ACTOR_ID).call(run_input=request)
    if not run or run.get("status") != "SUCCEEDED":
        raise RuntimeError(f"LIHKG Actor did not succeed: {run and run.get('status')}")
    rows = client.dataset(run["defaultDatasetId"]).list_items().items
    frame = normalize_social(rows, asset["ticker"], "lihkg", since=now - timedelta(days=lookback_days), until=datetime.now(timezone.utc))
    meta = {"actor_id": LIHKG_ACTOR_ID, "run_id": run.get("id"), "dataset_id": run.get("defaultDatasetId"), "retrieved_at": now.isoformat(), "count": len(frame), "actor_input": request}
    return frame, meta


def run_futu_community(asset: dict, *, enabled: bool = False) -> tuple[pd.DataFrame, dict]:
    """Official authenticated CLI (not an Apify Actor). Opt-in after rights review.

    Exact Futu JSON shape and per-post timestamp coverage MUST be checked on first run.
    No undocumented endpoint or bypass is used.
    """
    if not enabled:
        raise RuntimeError("Futu community access disabled pending terms/authorization review. Pass enabled=True only with permission.")
    cmd = ["futu", "search", asset["futu_symbol"], "--type", "community", "--format", "json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
    except FileNotFoundError as err:
        raise RuntimeError("Install and authorize official Futu CLI (futu config init) first") from err
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Official Futu CLI failed: {err.stderr[-400:]}") from err
    try:
        payload = json.loads(proc.stdout)
    except ValueError as err:
        raise RuntimeError("Futu CLI did not return JSON; inspect `futu search --help`") from err
    frame = normalize_social(payload, asset["ticker"], "futu")
    if frame.empty:
        raise RuntimeError("Futu response contained no parseable timestamped posts; inspect its schema. No rows were fabricated.")
    return frame, {"provider":"official_futu_cli", "command": cmd, "count": len(frame), "note":"Official CLI is NOT Apify-based; verify assignment accepts this or wrap an authorized integration in Apify."}


def fetch_exa_news(asset: dict, *, lookback_days: int = 10, max_results: int = 25, token: str | None = None) -> pd.DataFrame:
    from exa_py import Exa
    token = token or os.getenv("EXA_API_KEY")
    if not token:
        raise RuntimeError("Set EXA_API_KEY in .env")
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=lookback_days)
    query = f'"{asset["name"]}" Hong Kong stock earnings financial news HKEX'
    result = Exa(api_key=token).search(query, type="auto", num_results=max_results,
                                       start_published_date=start.strftime("%Y-%m-%d"),
                                       contents={"highlights": True})
    rows = []
    for hit in result.results:
        date = _timestamp(getattr(hit, "published_date", None))
        if date is None or date < _timestamp(start) or date > _timestamp(now):
            continue
        url = getattr(hit, "url", None)
        snippets = getattr(hit, "highlights", []) or []
        content = " ".join([str(getattr(hit, "title", "") or "")] + [str(v) for v in snippets]).strip()
        if not content or not url:
            continue
        rows.append({"ticker": asset["ticker"], "source": "exa_news", "item_id": url,
                     "url": url, "published_at": date, "text": content,
                     "retrieved_at": pd.Timestamp.now(tz="UTC")})
    return pd.DataFrame(rows, columns=NEWS_COLS).drop_duplicates("item_id") if rows else pd.DataFrame(columns=NEWS_COLS)


def fetch_daily_prices(asset: dict, *, start: str = "2025-01-01", end: str | None = None) -> pd.DataFrame:
    """Research-only prototype. Close is unadjusted, no licence to redistribute prices."""
    import yfinance as yf
    end = end or (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()
    bars = yf.Ticker(asset["ticker"]).history(start=start, end=end, interval="1d", auto_adjust=False)
    if bars is None or bars.empty:
        raise RuntimeError(f"No Yahoo historical price bars returned for {asset['ticker']}")
    out = bars.reset_index().rename(columns=str.lower)
    date_col = "date" if "date" in out else "datetime"
    out["session"] = pd.to_datetime(out[date_col]).dt.date.astype(str)
    cols = ["session", "open", "high", "low", "close", "volume"]
    missing = [x for x in cols if x not in out]
    if missing:
        raise RuntimeError(f"Missing Yahoo price fields: {missing}")
    out = out[cols].copy()
    out.insert(0, "ticker", asset["ticker"])
    return out.sort_values("session").reset_index(drop=True)


def write_frame(frame: pd.DataFrame, file_path: str | Path) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

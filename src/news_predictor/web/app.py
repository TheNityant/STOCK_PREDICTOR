"""FastAPI REST API + static reviewer-friendly dashboard (read-only by design)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from news_predictor.web.data import assets, get_asset, overview

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(
    title="HengSeng Sentiment Intelligence API",
    version="0.2.0",
    summary="Read-only research results and clearly labeled synthetic demo",
    description=("REST interface to the notebook-first research pipeline. No HTTP endpoint runs paid scraping, "
                 "training, third-party auth, trade execution, or untrusted model deserialization."),
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health", tags=["system"])
def health():
    return {"status": "ok", "service": "HengSeng Sentiment Intelligence", "version": app.version}


@app.get("/api/assets", tags=["assets"])
def list_assets():
    return {"assets": list(assets().values()), "count": len(assets())}


@app.get("/api/overview", tags=["research"])
def get_overview(
    mode: Literal["demo", "real"] = Query("demo"),
    sort_by: Literal["mentions", "sentiment", "direction"] = Query("mentions"),
):
    return overview(mode, sort_by)


@app.get("/api/assets/{ticker}/analysis", tags=["research"])
def get_analysis(ticker: str, mode: Literal["demo", "real"] = Query("demo")):
    if ticker not in assets():
        raise HTTPException(status_code=404, detail="Ticker not configured")
    result = get_asset(ticker, mode)
    if result is None:
        raise HTTPException(status_code=404, detail="No complete, scored local research dataset for this ticker. Run the notebook/CLI first, or select demo mode.")
    return result


@app.get("/api/sources", tags=["research"])
def get_sources():
    return {
        "real_collection_enabled_over_http": False,
        "sources": [
            {"name": "LIHKG", "type": "social", "integration": "Apify third-party Actor", "status": "adapter_written_live_access_unverified"},
            {"name": "Futu Community", "type": "social", "integration": "Official Futu CLI", "status": "disabled_pending_permission_schema_review"},
            {"name": "Exa", "type": "news", "integration": "Exa Python SDK", "status": "adapter_written_live_access_unverified"},
            {"name": "Yahoo Finance", "type": "price", "integration": "yfinance", "status": "research_only_not_official_exchange_feed"},
        ],
        "note": "Demo mode does not call any of these providers. Refer to docs/SOURCES.md for licenses and limitations.",
    }

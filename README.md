# NEWS_PREDICTOR — Hong Kong sentiment → next-close forecasting

**Notebook-first internship assessment project.** A Python workflow to collect social posts through Apify, retrieve Exa financial news, align publish timestamps to Hong Kong trading-session cutoffs, engineer sentiment and OHLCV features, train a next-session **return** regressor, evaluate on later unseen sessions, and produce an HTML/CSV report.

> **Status: working, tested codebase; NOT a pre-trained, proven stock predictor.** No API keys or live historical sentiment datasets were supplied. Training deliberately refuses to run on the 10-day-only social data if there are fewer than 60 fully observed daily examples. No results have been fabricated. Read [source and licensing review](docs/SOURCES.md).

## Web dashboard + REST API (new)

This project now also contains a **FastAPI REST API** and a single-page, responsive HTML/CSS/JavaScript dashboard. This is a *presentation layer* over the notebook — it does **not** replace the notebook's data acquisition, training, or evaluation stages.

**Try it without credentials** (synthetic demo is enabled by default):

```bash
python -m pip install -e ".[test]"
uvicorn news_predictor.web.app:app --reload --host 127.0.0.1 --port 8000
```

- Dashboard: http://127.0.0.1:8000/
- API interactive docs: http://127.0.0.1:8000/docs
- REST examples:

```bash
curl http://127.0.0.1:8000/api/health
curl 'http://127.0.0.1:8000/api/overview?mode=demo&sort_by=mentions'
curl 'http://127.0.0.1:8000/api/assets/0700.HK/analysis?mode=demo'
curl 'http://127.0.0.1:8000/api/overview?mode=real'
```

**Critical distinction:** Demo mode is wholly synthetic and fixed as of 2026-11-13; its mocked prices, sentiment, forecasts, signals and even example backtest metrics are fictional. They **must never be presented as live market data, real trained forecasts, or evidence of model accuracy**. Demo uses a deterministic seed and requires zero API keys. The in-app banner, forecast labels and evaluation cards carry this disclosure.

**Real-results mode** reads locally available `data/raw/<ticker>/prices.csv`, `social.csv`, `news.csv`, and optional `outputs/<ticker>/evaluation.json` written by your notebook/CLI. It never performs paid HTTP scraping, starts training, exposes raw posts or credentials, or deserializes `models.joblib` inside a public request. A live next-session forecast is deliberately unavailable until you explicitly implement a safe, validated inference-export workflow. Missing local files return an empty overview or a clear 404.

### Deploy on Railway or another container host

The `Dockerfile` runs the dashboard/API on `0.0.0.0:$PORT`; `/api/health` is the liveness check. From Railway, select **Deploy from GitHub repo** and choose this repo, then generate a public domain in Networking. With no secrets or datasets configured, the public site remains **synthetic demo only**. Railway may charge for service resources; review your plan before deploying. See [web deployment guide](docs/WEB_DEPLOY.md).

No database is required for the demo, no frontend build step is needed, and there are no POST endpoints that could trigger expenses or trades.

---

## Fast start

Python 3.10+ (3.12 recommended):

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e '.[test,sentiment]'
cp .env.example .env      # on Windows PowerShell: Copy-Item .env.example .env
# Fill APIFY_TOKEN and EXA_API_KEY in .env, never share it.
python -m pytest -q
jupyter notebook notebooks/01_hk_sentiment_forecasting.ipynb
```

Or use the terminal companion:

```bash
news-predictor collect --ticker 0700.HK --start 2025-01-01
news-predictor collect --ticker 9988.HK
news-predictor collect --ticker 1810.HK
news-predictor collect --ticker 3690.HK
news-predictor collect --ticker 1211.HK
news-predictor report               # no fabricated predictions if untrained
news-predictor train --ticker 0700.HK
news-predictor predict --ticker 0700.HK
```

**Futu is disabled by default.** After checking community data use rights, installing and authorizing the [official Futu CLI](https://open.futunn.com/cli/install), use `news-predictor collect --ticker 0700.HK --allow-futu`. If permission or parseable timestamped posts are unavailable, the run fails with an explanation. The official CLI is **not Apify**. As a result, the two-Apify-sources requirement is not yet proven compliant. See the source audit.

## Run stages in the notebook

1. Choose a ticker (`SELECTED_TICKER`) among five HKEX candidates in `config/assets.json`.
2. Explicitly enable live APIs in the notebook only after reviewing data source permissions and costs.
3. Collect 10-day social posts using LIHKG Apify Actor, recent Exa news, and separate daily OHLCV market prices. Optional Futu CLI source requires account authorization.
4. Review source coverage, deduplication and missing publication timestamps; label sentiment using a documented multilingual pretrained classifier.
5. Align by actual post publish time (UTC) to the first HK trading-session **16:00 HKT** cutoff at/after posting. Posts after close or over weekends shift to the next observed session.
6. Train price-only and price+sentiment models on data with both social sources present; evaluate chronologically with a one-row purge before an untouched later holdout. Report against the trivial **next close = current close** baseline.
7. Save `outputs/<ticker>/evaluation.json`, trusted local `models.joblib`, `report.html` and `outputs/exchange_report.html` + CSV leaderboard.
8. Use `news-predictor predict --ticker ...` for a trained ticker; if the latest session lacks both social sources, forecast availability is refused rather than fabricated.

## Expected input/output schemas

- `data/raw/<ticker>/prices.csv`: `ticker, session, open, high, low, close, volume`.
- `data/raw/<ticker>/social.csv`: `ticker, source, item_id, url, published_at, text, likes, reposts, views, author_followers, retrieved_at`, plus `sentiment` after scoring. Source values: `lihkg`, `futu`.
- `data/raw/<ticker>/news.csv`: `ticker, source, item_id, url, published_at, text, retrieved_at`, plus `sentiment` after scoring.
- `data/raw/<ticker>/provenance.json`: Actor ID, run ID, dataset ID, Actor inputs, counts and data-provider names (never API keys).
- `outputs/<ticker>/evaluation.json`: per-model held-out price MAE/RMSE, directional accuracy, return MAE, validation folds and actual-vs-predicted series.
- `outputs/<ticker>/report.html`: interactive actual-vs-predicted chart and model comparison.
- `outputs/exchange_report.html` and `.csv`: sorted ticker, company, last close, social mentions/sentiment, news sentiment, direction, position, support/resistance and forecast **only when a model exists**.

## Scientific scope, safeguards, and limitations

- **Not a chatbot / stock-link regressor:** symbol and keyword inputs identify data to collect. Model inputs are timestamp-aligned numerical features. We predict next-day return `P[t+1]/P[t]-1`, then derive `P[t+1]`.
- **Only genuine out-of-sample metrics.** Training examples require both social sources and a known next close; at least 60 sessions required (an exploratory engineering gate, not statistical proof of alpha).
- **Chronological splits:** 80/20-style final test tail and one supervised-row purge; internal TimeSeriesSplit when enough samples. No random splitting or trained-on-test accuracy.
- **Comparison:** same HistGradientBoostingRegressor architecture for price-only vs price+sentiment, plus a no-change baseline; no claims of profitable accuracy.
- **Missing metadata:** LIHKG actor cannot promise repost counts or author followers. Null is retained, not treated as zero; engagement aggregates use observed likes where present.
- **Important**: source 2 authorization, actor compliance, historical sentiment depth, professional market-data rights, HKT half-day calendar, non-English sentiment accuracy and quote/ticker relevance all need verification. This is a *research prototype* rather than production market intelligence.
- **Ticker list:** Tencent (0700.HK), Alibaba (9988.HK), Xiaomi (1810.HK), Meituan (3690.HK), BYD (1211.HK). Korea is mentioned in the assessment but exchange scope is not explicit; current implementation covers Hong Kong only.
- **Security:** `.env` and `data/raw` are ignored; never commit tokens, sourced posts, proprietary prices or untrusted joblib artifacts. Loading arbitrary `.joblib` objects is unsafe.

## Why the notebook plus Python files?

The `.ipynb` provides an easily reviewed end-to-end assessment walkthrough and human-readable charts. `src/news_predictor` contains unit-testable provider adapters, validation logic, and CLI workflow so we can reproduce analyses without hidden notebook state. A deployed website is not required.

## Technical references

See [docs/SOURCES.md](docs/SOURCES.md) for first-party Apify/Exa/Futu/sklearn/yfinance links, the third-party LIHKG Actor schema, license constraints and unresolved access blockers.

## Suggested next milestones

- Obtain permitted Futu discussion data and confirm an **Apify-based** path for the second source, or written approval for Futu CLI.
- Secure enough historical published-date-accurate sentiment for training and validation. If only ten days can be collected, submit a correctly labelled feasibility prototype rather than a false out-of-sample backtest.
- Replace regular close bucketing with HKEX trading calendar including half-days; audit multilingual sentiment on hand-labeled Chinese financial posts; validate data licences before sharing.

**For research/evaluation only. Not investment advice, trade execution, or a forecast guarantee.**

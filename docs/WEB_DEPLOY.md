# Web dashboard: API, frontend and deployment

## Architecture

- `src/news_predictor/web/app.py` — **read-only FastAPI** with automatic `/docs` OpenAPI documentation.
- `src/news_predictor/web/data.py` — deterministic synthetic demo and local artifact reader, **not** a training module.
- `src/news_predictor/web/static/index.html`, `styles.css`, `app.js` — zero-build responsive frontend; the browser fetches REST JSON and renders charts directly as SVG.
- Existing `notebooks/` and `src/news_predictor/{ingest,sentiment,features,model,report}.py` remain the source of truth for research workflows.

### Endpoint reference

| Method | Path | Response | API side effects |
|---|---|---|---|
| `GET` | `/` | Dashboard HTML | None |
| `GET` | `/api/health` | Health/version JSON | None |
| `GET` | `/api/assets` | Configured stock names and aliases | None |
| `GET` | `/api/overview?mode=demo&sort_by=mentions` | Synthetic/real leaderboard | None |
| `GET` | `/api/assets/0700.HK/analysis?mode=demo` | Chart series, sentiment, forecast and evaluation | None |
| `GET` | `/api/sources` | Source integration status; **not** tokens | None |
| `GET` | `/docs` | Interactive Swagger/OpenAPI | None |

`mode` accepts `demo|real`. `sort_by` accepts `mentions|sentiment|direction`. All other choices return HTTP 422, unknown ticker returns 404. The frontend starts in `demo` so first-time reviewers can inspect the UI without having API keys or raw data.

### Demonstration vs real research

`demo` creates 70 synthetic price sessions and fictional LIHKG/Futu/Exa-like counts and synthetic illustration backtest figures using a fixed seed. No scraped post is copied into the demo. As-of date is hardcoded (2026-11-13) and not a live feed. The UI labels everything **Synthetic demo** and warns that backtests/forecasts are **not** trained market results.

`real` reads local CSVs previously generated through notebook/CLI; no HTTP endpoint can fetch data, spend paid API credits, retrain models or execute trading instructions. Historical backtest metrics are displayed only when a valid `evaluation.json` exists. The server does **not** unpickle model files; the latest future price remains unavailable until a reviewed export mechanism is built. To expose actual results on a deployed service, make a deliberate data-distribution decision: do not commit licensed raw posts/quotes or secrets into a public repo.

### Local run

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
uvicorn news_predictor.web.app:app --reload --port 8000
```

Open http://127.0.0.1:8000 and http://127.0.0.1:8000/docs.

### Railway deployment

1. Connect your GitHub account to Railway and create a project from `TheNityant/STOCK_PREDICTOR` (only if you accept Railway's pricing).
2. Railway detects the root `Dockerfile` and `railway.json`; the service listens on `$PORT` and has a health check at `/api/health`.
3. From service Settings → Networking, **Generate Domain**. A repository URL is not itself a deployed application URL.
4. The URL serves demo mode without API keys. Verify `/api/health`, `/docs` and the 5-asset synthetic leaderboard.
5. Share the URL with a clear demo-only disclosure alongside the GitHub repository. For authentic empirical results, run notebook research separately and provide a verified report with source rights respected.

### Safety and limitations

- No API keys, private forum text, personal information or licensed quotes are shipped in Docker builds (see `.dockerignore`).
- HTTP routes only permit known configured tickers; no arbitrary filesystem access or POST scraping.
- The front end renders user-controlled text using HTML escaping.
- Public mode deliberately avoids `joblib.load` because model artifact deserialization can execute code.
- No authentication or rate limiting is needed for the read-only demo with no sensitive data, but would be required before adding expensive collection or mutable endpoints.
- Source permissions, actual trained metrics, Hong Kong market calendar and second approved Apify source remain independently unresolved in the internship assessment.

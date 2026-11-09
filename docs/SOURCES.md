# Source provenance and rights review (November 2026)

## First-party documentation and exact dependencies

| Component | Documentation | Implementation | Caveat |
| --- | --- | --- | --- |
| Apify | https://docs.apify.com/api/client/python | `ApifyClient.actor().call()` and dataset items | API token required, Actor usage may incur charges |
| LIHKG Apify Actor | https://apify.com/claude_code_reviewer/lihkg-scraper-en | `mode=search`, `keywords`, `dateFrom`, `desc_create_time` | Community actor **not first-party LIHKG**; schema not independently tested; verify terms and rights before running |
| Futu official CLI | https://open.futunn.com/cli/research and https://open.futunn.com/cli/install | `futu search HK.00700 --type community --format json` | Must authenticate, inspect return schema, and verify permissible reuse; **not Apify-based**, unlike the assignment's preferred ingestion method |
| Exa official Python SDK | https://github.com/exa-labs/exa-py | `Exa.search` with publication-date filter and highlights | Exa index time is not publish time. Unknown `published_date` rows rejected |
| Research price feed | https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html | `Ticker.history(auto_adjust=False,interval="1d")` | yfinance disclaimer: personal/research use; not official HKEX pricing or licensed redistribution |
| Multilingual sentiment model | https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student | 3-class sentiment score = P(positive)-P(negative) | General multilingual, **not calibrated to HK financial slang**; sample labeling audit required |
| Time-series validation | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html | Chronological holdout + one-row purge + rolling CV | Need genuinely long sentiment history for defensible model validation |

## Unresolved blocker list

1. No verified authorized Futu **Apify Actor** is available in this repository; the Futu CLI is an official fallback, not evidence that both sources were scraped via Apify. Ask recruiter if a second Apify integration is mandatory.
2. Futu community output structure, history depth, metadata fields and permitted data use must be empirically checked. The implementation rejects unsupported shapes rather than guesses.
3. LIHKG Actor is third-party despite running on Apify. Its `search` mode returns **thread results**, not a guaranteed full reply feed. Deeper `thread` runs would be required for replies, subject to expense and rights review.
4. Only 10 days of social records does **not** make a valid ~60-session daily sentiment model. The training gate is intentionally strict.
5. Generic 16:00 HKT close bucketing does not account for half-days, trading halts, or emergency closures. Prices come from the actual observed session list, but an exchange calendar must be integrated for a production backtest.
6. No downloaded prices, trained weights, real API responses, or achieved validation metrics are included in the repository. Tests use **synthetic fixtures strictly for correctness testing**.
7. No source allows the assistant to assert commercial training rights automatically. Confirm each publisher's terms, especially before making data public or using for training.

## Data-handling policy

- Treat all post timestamps as publication timestamps converted to UTC; `scraped_at` is never the original posting time.
- Reject news without credible published dates. Do not forward-fill sentiment to earlier sessions.
- Do not put Apify/Exa/Futu credentials in code, notebooks, Git history, or CSV exports.
- Do not upload raw forum posts or licensed market datasets into public GitHub without redistribution rights.
- Fit sentiment model (pretrained classifier) independently of the price-return model (locally trained regression).

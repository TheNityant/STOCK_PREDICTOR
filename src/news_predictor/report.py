"""HTML report written from REAL evaluated rows; no simulated numbers."""
from __future__ import annotations
import html
from pathlib import Path
import pandas as pd


def render_html(evaluation: dict, ticker: str, path: str | Path) -> Path:
    import plotly.graph_objects as go
    from plotly.offline import plot
    x = evaluation["test_sessions"]
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=x, y=evaluation["actual_prices"], name="Actual next close", mode="lines+markers"))
    for name in ("price_only", "price_plus_sentiment"):
        chart.add_trace(go.Scatter(x=x, y=evaluation["models"][name]["predicted_prices"], name=name, mode="lines+markers"))
    chart.update_layout(title=f"{ticker} — strictly held-out next-close predictions", xaxis_title="Feature session", yaxis_title="Next close (HKD)")
    fig_html = plot(chart, include_plotlyjs=True, output_type="div")
    table = pd.DataFrame([{"model":name, **values["test"]} for name,values in evaluation["models"].items()]).to_html(index=False, border=0, float_format=lambda x:f"{x:.5f}")
    content = f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(ticker)} evaluation</title><style>body{{font-family:system-ui;max-width:1100px;margin:35px auto;padding:0 15px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style></head><body><h1>{html.escape(ticker)} — Sentiment Price Prediction</h1><p>Research-only. The test set occurs after training; reported metrics do not establish profitable forecasting.</p>{fig_html}<h2>Held-out performance</h2>{table}<p>Train ends: {html.escape(evaluation['train_last_session'])}. Test starts: {html.escape(evaluation['test_first_session'])}. One labeled training row is purged across the split boundary.</p></body></html>"
    p = Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content,encoding="utf-8");return p


def summarize_ticker(asset: dict, frame: pd.DataFrame, trained: dict | None = None) -> dict:
    """No model forecast or position unless a legitimately trained model is present."""
    from .model import predict_next
    last = frame.iloc[-1]
    response = {
        "exchange":asset["exchange"], "ticker":asset["ticker"], "asset_name":asset["name"],
        "as_of_session":str(last.session), "current_close":float(last.close),
        "social_mentions":int(last.total_social_mentions),
        "social_sentiment":None if pd.isna(last.social_sentiment) else float(last.social_sentiment),
        "news_sentiment":None if pd.isna(last.news_sentiment) else float(last.news_sentiment),
        "support":None if pd.isna(last.support) else float(last.support),
        "resistance":None if pd.isna(last.resistance) else float(last.resistance),
        "predicted_next_close":None,"direction":"unavailable","position":"no_position",
        "prediction_status":"not trained or current two-source coverage unavailable"
    }
    if trained is not None and int(last.source_count) == 2:
        response.update(predict_next(last, trained))
        response["prediction_status"]="trained model (research only)"
    return response


def render_leaderboard(rows: list[dict], path: str | Path, sort_by: str = "social_mentions") -> Path:
    """Per-exchange, sortable visual HTML table; no artificial forecast placeholders."""
    import plotly.express as px
    from plotly.offline import plot
    if not rows:
        raise ValueError("No ticker observations to report")
    allowed={"social_mentions","social_sentiment","direction"}
    if sort_by not in allowed:
        raise ValueError(f"Sort by one of {allowed}")
    df=pd.DataFrame(rows).sort_values(["exchange",sort_by],ascending=[True,False],na_position="last")
    fig=px.bar(df, x="ticker", y="social_mentions", color="exchange", hover_data=["asset_name", "social_sentiment"], title="Social mentions by ticker")
    chart=plot(fig,include_plotlyjs=True,output_type="div")
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    body="<h1>Stock sentiment and next-period forecast</h1><p>Only observed values. Unavailable predictions are not estimated. Research only.</p>"
    for exchange, section in df.groupby("exchange"):
        body += f"<h2>{html.escape(str(exchange))}</h2>"+section.drop(columns="exchange").to_html(index=False,na_rep="Unavailable",float_format=lambda x:f"{x:.4f}")
    p.write_text("<!doctype html><html><head><meta charset='utf-8'><title>Exchange sentiment</title><style>body{font-family:system-ui;max-width:1350px;margin:30px auto;padding:0 18px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}th{background:#eee}</style></head><body>"+body+chart+"</body></html>",encoding="utf-8")
    df.to_csv(p.with_suffix(".csv"),index=False)
    return p

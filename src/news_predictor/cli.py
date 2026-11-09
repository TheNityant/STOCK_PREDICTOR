"""CLI companion for the notebook; no deployed server is required."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from .ingest import fetch_daily_prices, fetch_exa_news, run_lihkg_actor, run_futu_community, write_frame
from .sentiment import label_sentiment
from .features import build_features
from .model import train_evaluate, save_artifacts, predict_next
from .report import render_html, summarize_ticker, render_leaderboard


def assets_by_ticker(path="config/assets.json"):
    return {x["ticker"]:x for x in json.loads(Path(path).read_text(encoding="utf-8"))}


def _load(folder:Path, name:str) -> pd.DataFrame:
    path=folder/name
    if not path.exists():
        raise FileNotFoundError(f"Expected {path}. Run `collect` first; no sample data substituted.")
    return pd.read_csv(path)


def main(argv=None):
    load_dotenv()
    p=argparse.ArgumentParser(description="Hong Kong sentiment forecast research (not financial advice)")
    p.add_argument("command",choices=["collect","train","predict","report"])
    p.add_argument("--ticker",default="0700.HK")
    p.add_argument("--assets",default="config/assets.json")
    p.add_argument("--data",default="data/raw")
    p.add_argument("--outputs",default="outputs")
    p.add_argument("--start",default="2025-01-01",help="Earliest daily price history date")
    p.add_argument("--allow-futu",action="store_true",help="Use authenticated official Futu CLI after checking data use permission")
    p.add_argument("--min-rows",type=int,default=60)
    args=p.parse_args(argv)
    assets=assets_by_ticker(args.assets)
    if args.ticker not in assets:
        p.error(f"Unknown ticker {args.ticker}, choose from {list(assets)}")
    asset=assets[args.ticker]; path=Path(args.data)/args.ticker; out=Path(args.outputs)/args.ticker
    if args.command == "report":
        rows = []
        for ticker, each in assets.items():
            folder = Path(args.data)/ticker
            if not (folder/"prices.csv").exists():
                print(f"Skipped {ticker}: no collected data")
                continue
            pframe = _load(folder,"prices.csv")
            sframe = _load(folder,"social.csv")
            nframe = _load(folder,"news.csv")
            if "sentiment" not in sframe:
                sframe = label_sentiment(sframe)
            if "sentiment" not in nframe:
                nframe = label_sentiment(nframe)
            featured = build_features(pframe, sframe, nframe)
            model_file = Path(args.outputs)/ticker/"models.joblib"
            trained = None
            if model_file.exists():
                import joblib
                # Load only artifacts produced locally; pickle/joblib loads can run code.
                trained = joblib.load(model_file)["price_plus_sentiment"]
            rows.append(summarize_ticker(each,featured,trained))
        if not rows:
            raise RuntimeError("No collected ticker data. Run `collect` on at least one ticker first.")
        target = render_leaderboard(rows,Path(args.outputs)/"exchange_report.html")
        print(f"Report: {target}; CSV: {target.with_suffix('.csv')}")
        return
    if args.command=="collect":
        prices=fetch_daily_prices(asset,start=args.start)
        lihkg,meta=run_lihkg_actor(asset)
        futu, fmeta = run_futu_community(asset,enabled=args.allow_futu) if args.allow_futu else (pd.DataFrame(columns=lihkg.columns),{"status":"disabled_not_authorized"})
        news=fetch_exa_news(asset)
        if lihkg.empty: raise RuntimeError("LIHKG produced no valid timestamped posts; inspect Actor run")
        social=pd.concat([lihkg,futu],ignore_index=True)
        write_frame(prices,path/"prices.csv");write_frame(social,path/"social.csv");write_frame(news,path/"news.csv")
        (path/"provenance.json").write_text(json.dumps({"lihkg":meta,"futu":fmeta,"exa": {"source":"exa_py", "count":len(news)},"prices":{"source":"yfinance","count":len(prices)}},indent=2),encoding="utf-8")
        print(f"Collected prices={len(prices)} social={len(social)} news={len(news)} to {path}. Futu enabled: {args.allow_futu}")
    else:
        prices=_load(path,"prices.csv");social=_load(path,"social.csv");news=_load(path,"news.csv")
        # Explicit sentiment labeling; may download pretrained model on first use.
        for kind,df in (("social",social),("news",news)):
            if "sentiment" not in df.columns:
                if df.empty:
                    df["sentiment"]=pd.Series(dtype=float)
                else:
                    df=label_sentiment(df)
                write_frame(df,path/f"{kind}.csv")
            if kind=="social": social=df
            else:news=df
        frame=build_features(prices,social,news)
        if args.command=="train":
            evaluation, models = train_evaluate(frame,min_rows=args.min_rows)
            save_artifacts(evaluation,models,out)
            render_html(evaluation,args.ticker,out/"report.html")
            print(json.dumps({"ticker":args.ticker,"test_metrics":{k:v["test"] for k,v in evaluation["models"].items()},"report":str(out/"report.html")},indent=2))
        else:
            import joblib
            model_path=out/"models.joblib"
            if not model_path.exists():raise FileNotFoundError("No trained model. Run `train` first using sufficient historical sentiment.")
            # SECURITY: Only load model artifacts produced locally and trusted by you.
            models=joblib.load(model_path)
            last=frame.iloc[-1]
            print(json.dumps(predict_next(last,models["price_plus_sentiment"]),indent=2))

if __name__=="__main__":main()

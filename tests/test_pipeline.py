from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from news_predictor.ingest import normalize_social, _records
from news_predictor.sentiment import label_sentiment
from news_predictor.features import build_features, eligible_training_rows
from news_predictor.model import train_evaluate, predict_next
from news_predictor.report import summarize_ticker, render_leaderboard


def prices(sessions=150):
    days = pd.bdate_range("2026-01-05",periods=sessions)
    close = 100 + 0.12*np.arange(sessions) + 2*np.sin(np.arange(sessions)/7)
    return pd.DataFrame({"ticker":"0700.HK", "session":days.strftime("%Y-%m-%d"),
                         "open":close-0.2,"high":close+1.3,"low":close-1.2,
                         "close":close,"volume":1_000_000+10000*np.arange(sessions)})


def observed_data(n=150):
    p = prices(n)
    cutoffs = pd.to_datetime(p.session).dt.tz_localize("Asia/Hong_Kong").add(pd.Timedelta(hours=16)).dt.tz_convert("UTC")
    rows=[]
    for i,cut in enumerate(cutoffs):
        for source in ("lihkg","futu"):
            rows.append({"ticker":"0700.HK","source":source,"item_id":f"{source}-{i}",
                         "published_at":cut-pd.Timedelta(hours=1),"text":"genuine (synthetic test)",
                         "likes":i%6, "sentiment":float(np.sin(i/5))})
    s=pd.DataFrame(rows)
    news = pd.DataFrame([{"ticker":"0700.HK","source":"exa_news","item_id":f"news-{i}",
                          "published_at":cut-pd.Timedelta(hours=2),"text":"example",
                          "sentiment":0.4} for i,cut in enumerate(cutoffs) if i%3==0])
    return p,s,news


def test_normalizer_uses_post_time_not_scrape_time():
    raw=[{"thread_id":"12","created_at":"2026-02-03T12:00:00+08:00",
          "title":"騰訊","own_text":"看好","like_count":4},
         {"thread_id":"12","created_at":"2026-02-03T12:00:00+08:00","title":"dup"},
         {"thread_id":"13","scraped_at":"2026-02-03T12:00:00Z","text":"no publish time"}]
    df=normalize_social(raw,"0700.HK","lihkg")
    assert len(df)==1
    assert df.iloc[0].published_at==pd.Timestamp("2026-02-03T04:00:00Z")
    assert df.iloc[0].likes==4
    assert pd.isna(df.iloc[0].author_followers)


def test_unrecognized_futu_shape_fails_closed():
    with pytest.raises(ValueError,match="recognized"):
        _records({"unexpected_key":{}})


def test_sentiment_requires_meaningful_labels():
    class Dummy:
        def __call__(self,texts,**kwargs):return [[{"label":"positive","score":0.8},{"label":"negative","score":0.1},{"label":"neutral","score":0.1}] for t in texts]
    scored=label_sentiment(pd.DataFrame({"text":["excellent"]}),classifier=Dummy())
    assert scored.sentiment.iloc[0]==pytest.approx(0.7)


def test_post_after_close_moves_to_next_session():
    p=prices(23)
    day0=pd.Timestamp(p.iloc[-2].session,tz="Asia/Hong_Kong")
    s=pd.DataFrame([{"ticker":"0700.HK","source":"lihkg","item_id":"a","published_at":day0+pd.Timedelta(hours=17),"text":"late","likes":0,"sentiment":1.0}])
    features=build_features(p,s,pd.DataFrame())
    assert features.iloc[-2].lihkg_mentions==0
    assert features.iloc[-1].lihkg_mentions==1


def test_future_post_never_leaks_to_earlier_session():
    p=prices(23)
    future=pd.Timestamp(p.iloc[-1].session,tz="Asia/Hong_Kong")+pd.Timedelta(days=3)
    s=pd.DataFrame([{"ticker":"0700.HK","source":"lihkg","item_id":"a","published_at":future,"text":"future","likes":0,"sentiment":1.0}])
    features=build_features(p,s,pd.DataFrame())
    assert features.lihkg_mentions.sum()==0


def test_short_history_fails_instead_of_faking_accuracy():
    p,s,n=observed_data(22)
    f=build_features(p,s,n)
    with pytest.raises(ValueError,match="Only"):
        eligible_training_rows(f,min_rows=60)


def test_full_chronological_train_test_and_prediction(tmp_path):
    p,s,n=observed_data()
    features=build_features(p,s,n)
    metrics, models=train_evaluate(features)
    assert metrics["test_first_session"]>metrics["train_last_session"]
    assert metrics["purge_sessions"]==1
    assert metrics["models"]["price_plus_sentiment"]["test"]["n"] >= 10
    forecast=predict_next(features.iloc[-1],models["price_plus_sentiment"])
    assert forecast["ticker"]=="0700.HK"
    assert forecast["predicted_next_close"]>0
    asset={"exchange":"HKEX","ticker":"0700.HK","name":"Tencent Holdings"}
    data=summarize_ticker(asset,features,models["price_plus_sentiment"])
    assert data["direction"] in ("up","down","flat")
    report=render_leaderboard([data],tmp_path/"leaderboard.html")
    assert report.exists() and report.with_suffix(".csv").exists()


def test_no_model_means_unavailable_forecast():
    p,s,n=observed_data(23)
    f=build_features(p,s,n)
    asset={"exchange":"HKEX","ticker":"0700.HK","name":"Tencent"}
    summary=summarize_ticker(asset,f)
    assert summary["predicted_next_close"] is None
    assert summary["direction"]=="unavailable"

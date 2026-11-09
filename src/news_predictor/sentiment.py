"""Multilingual general sentiment scoring; not yet validated for HK stock jargon.

A sentiment classifier is an imported pretrained text model. The financial return
regressor is the model trained in this project. Never fake missing sentiment scores.
"""
from __future__ import annotations
import pandas as pd

MODEL_ID = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"


def label_sentiment(frame: pd.DataFrame, *, batch_size: int = 16, classifier=None) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        result["sentiment"] = pd.Series(dtype=float)
        return result
    if "text" not in result.columns:
        raise ValueError("Expected a text field")
    if classifier is None:
        try:
            from transformers import pipeline
        except ImportError as e:
            raise RuntimeError('Install NLP dependencies: pip install -e ".[sentiment]"') from e
        classifier = pipeline("text-classification", model=MODEL_ID, top_k=None, truncation=True)
    texts = result["text"].fillna("").astype(str).tolist()
    predictions = classifier(texts, batch_size=batch_size, truncation=True)
    if len(predictions) != len(result):
        raise RuntimeError("Sentiment model returned wrong number of labels")
    scores = []
    for record in predictions:
        if isinstance(record, dict):
            record = [record]
        probs = {str(p["label"]).lower(): float(p["score"]) for p in record}
        if not ("positive" in probs and "negative" in probs):
            raise ValueError(f"Unexpected sentiment labels: {list(probs)}")
        scores.append(probs["positive"] - probs["negative"])
    result["sentiment"] = scores
    result["sentiment_model"] = MODEL_ID
    return result

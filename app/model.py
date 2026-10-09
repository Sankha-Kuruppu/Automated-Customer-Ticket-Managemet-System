"""Loads the trained model bundle and turns raw probabilities into contract-compliant predictions."""
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np

from app.labels import CATEGORY_TO_TEAM, NO_SECONDARY
from app.preprocess import build_input, encoder_input

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "ticket_model.joblib"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decide(bundle: dict, p_cat: np.ndarray, p_sec: np.ndarray, p_urg: np.ndarray) -> list[dict]:
    """Apply thresholds and the consistency rules to probability matrices (one row per ticket).

    Shared by serving and by training-time evaluation so reported scores match the API exactly.
    """
    cat_classes = bundle["category_classes"]
    sec_classes = bundle["secondary_classes"]
    sec_none = sec_classes.index(NO_SECONDARY)
    sec_threshold = bundle["secondary_threshold"]
    urg_threshold = bundle["urgency_threshold"]
    review_threshold = bundle["review_threshold"]

    out = []
    for i in range(p_cat.shape[0]):
        ci = int(np.argmax(p_cat[i]))
        category = cat_classes[ci]
        confidence = float(p_cat[i, ci])

        # Secondary: best non-primary, non-"none" class, kept only if probable enough.
        scores = p_sec[i].copy()
        scores[sec_none] = -1.0
        if category in sec_classes:
            scores[sec_classes.index(category)] = -1.0
        si = int(np.argmax(scores))
        secondary = sec_classes[si] if scores[si] >= sec_threshold else None

        is_urgent = bool(p_urg[i] >= urg_threshold)

        # Contract rule: spam is never urgent and never has a secondary label.
        if category == "spam_irrelevant":
            is_urgent = False
            secondary = None
        if secondary == "spam_irrelevant":
            secondary = None

        out.append(
            {
                "category": category,
                "secondary_category": secondary,
                "team": CATEGORY_TO_TEAM[category],
                "is_urgent": is_urgent,
                "confidence": round(min(max(confidence, 0.0), 1.0), 4),
                "needs_human_review": confidence < review_threshold,
                "urgency_score": round(float(p_urg[i]), 4),
            }
        )
    return out


def head_proba(head: dict, X_tfidf, X_emb) -> np.ndarray:
    """Weighted soft-vote of the TF-IDF model and the embedding model for one prediction head."""
    w = head["weight_emb"]
    p = (1.0 - w) * head["tfidf"].predict_proba(X_tfidf)
    if w > 0:
        p = p + w * head["emb"].predict_proba(X_emb)
    return apply_temperature(p, head.get("temperature", 1.0))


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    """Temperature scaling on probabilities (t > 1 softens, t < 1 sharpens); rows still sum to 1."""
    if t == 1.0:
        return p
    q = np.power(np.clip(p, 1e-12, 1.0), 1.0 / t)
    return q / q.sum(axis=1, keepdims=True)


def predict_features(bundle: dict, X_tfidf, X_emb) -> list[dict]:
    heads = bundle["heads"]
    p_cat = head_proba(heads["category"], X_tfidf, X_emb)
    p_sec = head_proba(heads["secondary"], X_tfidf, X_emb)
    p_urg = head_proba(heads["urgency"], X_tfidf, X_emb)[:, 1]
    return decide(bundle, p_cat, p_sec, p_urg)


class TicketClassifier:
    def __init__(self, path: Path = DEFAULT_MODEL_PATH, encoder=None):
        self.path = Path(path)
        self.bundle = joblib.load(self.path)
        self.model_version: str = self.bundle["model_version"]
        self.sha256 = file_sha256(self.path)
        if encoder is None:
            from app.encoder import Encoder
            encoder = Encoder()
        self.encoder = encoder

    def predict(self, tickets: list[dict]) -> list[dict]:
        """tickets: dicts with channel, subject (optional) and text. Returns one result per ticket."""
        texts = [build_input(t["channel"], t.get("subject"), t["text"]) for t in tickets]
        X_tfidf = self.bundle["vectorizer"].transform(texts)
        X_emb = self.encoder.encode([encoder_input(t.get("subject"), t["text"]) for t in tickets])
        results = predict_features(self.bundle, X_tfidf, X_emb)
        for r in results:
            r.pop("urgency_score", None)
            r["model_version"] = self.model_version
        return results

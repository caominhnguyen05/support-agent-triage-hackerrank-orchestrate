from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import re
from rules import TRAINING_DATA, BUG_SIGNALS, INVALID_SIGNALS, FEATURE_REQUEST_SIGNALS, PRODUCT_AREA_RULES

class DomainClassifier:
    """
    TF-IDF + Logistic Regression classifier for classifying 
    tickets to domains: hackerrank | claude | visa | general

    Trains automatically on construction using TRAINING_DATA.
    """

    CONFIDENCE_THRESHOLD = 0.45

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,     
        )
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.5,               
        )
        self.is_trained = False
        self._auto_train()

    def _auto_train(self):
        texts  = [t for t, _ in TRAINING_DATA]
        labels = [l for _, l in TRAINING_DATA]
        self.train(texts, labels)

    def train(self, texts: list[str], labels: list[str]) -> None:
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_trained = True

    def retrain(self, extra_texts: list[str], extra_labels: list[str]) -> None:
        """Merge extra examples with the base training set and retrain."""
        base_texts  = [t for t, _ in TRAINING_DATA]
        base_labels = [l for _, l in TRAINING_DATA]
        self.train(base_texts + extra_texts, base_labels + extra_labels)

    def predict(self, text: str) -> tuple[str, float]:
        """
        Returns (domain, confidence).
        If confidence is below CONFIDENCE_THRESHOLD, returns ("uncertain", confidence)
        so the agent can fall back to keyword matching.
        """
        if not self.is_trained:
            return "uncertain", 0.0

        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        idx = int(np.argmax(probs))
        domain = self.model.classes_[idx]
        confidence = float(probs[idx])

        if confidence < self.CONFIDENCE_THRESHOLD:
            return "uncertain", confidence

        return domain, confidence

def _word_match(keyword: str, text: str) -> bool:
    if " " in keyword:
        return keyword in text
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text))

def classify_product_area(domain: str, issue: str, subject: str) -> str | None:
    combined = f"{subject or ''} {issue}".lower()
    scores = {}

    for rule_domain, keywords, label in PRODUCT_AREA_RULES:
        if rule_domain not in (domain, "general"):
            continue

        matches = sum(_word_match(kw, combined) for kw in keywords)
        if matches > 0:
            scores[label] = scores.get(label, 0) + matches

    if scores:
        return max(scores, key=scores.get)

def classify_request_type(issue: str, subject: str) -> str | None:
    combined = f"{subject or ''} {issue}".lower()
    if any(_word_match(sig, combined) for sig in INVALID_SIGNALS):
        return "invalid"
    if any(_word_match(sig, combined) for sig in BUG_SIGNALS):
        return "bug"
    if any(_word_match(sig, combined) for sig in FEATURE_REQUEST_SIGNALS):
        return "feature_request"
    return None
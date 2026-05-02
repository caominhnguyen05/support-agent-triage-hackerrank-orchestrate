from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np


TRAINING_DATA = [
    # hackerrank
    ("my coding test link expired", "hackerrank"),
    ("the assessment won't load in my browser", "hackerrank"),
    ("I cannot submit my solution", "hackerrank"),
    ("the proctoring webcam is not working", "hackerrank"),
    ("recruiter sent me a broken test link", "hackerrank"),
    ("my hackerrank score is wrong", "hackerrank"),
    ("interview challenge is showing an error", "hackerrank"),
    ("coding challenge time ran out too early", "hackerrank"),
    ("I failed the hiring assessment unfairly", "hackerrank"),
    ("test environment keeps crashing", "hackerrank"),
    ("IDE inside hackerrank is broken", "hackerrank"),
    ("my candidate report is missing", "hackerrank"),
    ("plagiarism flag on my submission is incorrect", "hackerrank"),
    ("remote proctoring disconnected mid-test", "hackerrank"),
    ("cannot access my hackerrank dashboard", "hackerrank"),

    # claude
    ("claude is not responding to my messages", "claude"),
    ("my claude pro subscription was charged twice", "claude"),
    ("how do I cancel my claude.ai plan", "claude"),
    ("artifacts are not rendering in the conversation", "claude"),
    ("context window limit reached too quickly", "claude"),
    ("I lost my conversation history in claude", "claude"),
    ("anthropic charged me but I cannot access pro", "claude"),
    ("claude is giving wrong answers about recent events", "claude"),
    ("how do I export my claude conversations", "claude"),
    ("claude keeps forgetting things mid chat", "claude"),
    ("my claude account is locked", "claude"),
    ("claude api rate limit exceeded", "claude"),
    ("image upload is not working in claude", "claude"),
    ("claude is much slower than usual today", "claude"),
    ("team plan seat not showing up for my colleague", "claude"),

    # visa
    ("my visa card was declined at the supermarket", "visa"),
    ("international transaction fee on my statement", "visa"),
    ("I did not make this visa purchase", "visa"),
    ("ATM would not give me cash with my visa card", "visa"),
    ("how do I dispute a charge on my visa card", "visa"),
    ("contactless payment not working on my card", "visa"),
    ("visa card expired and new one not arrived", "visa"),
    ("my card PIN is blocked after wrong attempts", "visa"),
    ("merchant charged me twice on visa", "visa"),
    ("online payment declined but card is valid", "visa"),
    ("visa virtual card not accepted by merchant", "visa"),
    ("foreign currency conversion rate seems wrong", "visa"),
    ("card transaction pending for too many days", "visa"),
    ("lost my visa card abroad what do I do", "visa"),
    ("cvv not accepted during checkout", "visa"),

    # general
    ("hello I have a question", "general"),
    ("please help me with my account", "general"),
    ("I need support urgently", "general"),
    ("something is not working properly", "general"),
    ("can someone assist me please", "general"),
]


class DomainClassifier:
    """
    TF-IDF + Logistic Regression classifier for routing tickets
    to: hackerrank | claude | visa | general

    Trains automatically on construction using TRAINING_DATA.
    Call retrain(texts, labels) to extend with domain-specific examples.
    """

    CONFIDENCE_THRESHOLD = 0.45

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,        # dampens high-frequency terms
        )
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.5,                    # slight regularisation increase for small corpus
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
        so the caller can fall back to keyword matching.
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
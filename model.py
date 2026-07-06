"""
model.py — trains a lightweight TF-IDF + Logistic Regression classifier
on labeled scam/legit text, and blends its probability with the
rule-based flag score from flags.py to produce a final verdict.

Train once (creates scam_model.joblib), then reuse via ScamDetector.
"""

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from flags import rule_score, detect_flags
from data.sample_data import SCAM_EXAMPLES, LEGIT_EXAMPLES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scam_model.joblib")

# How much weight the trained ML model vs the rule engine gets in the
# final blended score. Rules are hand-crafted and precise; the ML model
# generalizes to phrasing the rules don't cover.
ML_WEIGHT = 0.55
RULE_WEIGHT = 0.45


def train_and_save(path: str = MODEL_PATH) -> Pipeline:
    texts = SCAM_EXAMPLES + LEGIT_EXAMPLES
    labels = [1] * len(SCAM_EXAMPLES) + [0] * len(LEGIT_EXAMPLES)  # 1 = scam

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(texts, labels)
    joblib.dump(pipe, path)
    return pipe


def load_model(path: str = MODEL_PATH) -> Pipeline:
    if not os.path.exists(path):
        return train_and_save(path)
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Error loading model from {path} (possibly due to version mismatch): {e}")
        print("Retraining model locally...")
        return train_and_save(path)


class ScamDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        self.pipe = load_model(model_path)

    def analyze(self, text: str) -> dict:
        text = text.strip()
        if not text:
            return {
                "verdict": "UNKNOWN",
                "trust_score": None,
                "scam_probability": None,
                "reasons": [],
            }

        ml_prob = float(self.pipe.predict_proba([text])[0][1])  # P(scam)
        r_score = rule_score(text)
        hits = detect_flags(text)

        blended = ML_WEIGHT * ml_prob + RULE_WEIGHT * r_score

        # A single very strong rule (credential/payment/remote-access request)
        # should never be fully washed out by a low ML score.
        strong_hits = [f for f, _ in hits if f.weight >= 0.65]
        if strong_hits:
            blended = max(blended, 0.7)

        if blended >= 0.65:
            verdict = "LIKELY SCAM"
        elif blended >= 0.35:
            verdict = "SUSPICIOUS — proceed with caution"
        else:
            verdict = "LIKELY SAFE"

        trust_score = round((1 - blended) * 100, 1)  # 0-100, higher = more trustworthy

        reasons = [
            {"flag": f.name, "matched": snippet, "why": f.explanation}
            for f, snippet in hits
        ]

        return {
            "verdict": verdict,
            "trust_score": trust_score,          # 0 (no trust) - 100 (fully trustworthy)
            "scam_probability": round(blended * 100, 1),  # 0-100
            "ml_probability": round(ml_prob * 100, 1),
            "rule_score": round(r_score * 100, 1),
            "reasons": reasons,
        }


if __name__ == "__main__":
    train_and_save()
    print(f"Model trained and saved to {MODEL_PATH}")

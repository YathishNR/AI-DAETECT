import os
import random
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from config import MODEL_PATH
from modules.preprocessor import clean_text


SAFE_SAMPLES = [
    "The quarterly revenue increased by five percent year over year.",
    "Meeting scheduled for next Tuesday at the downtown office.",
    "Please review the attached document and share feedback.",
    "Weather forecast predicts light rain over the weekend.",
    "The product launch timeline has been updated on the wiki.",
    "Team lunch is planned for Friday at noon.",
    "Infrastructure upgrade completed successfully last night.",
    "Customer satisfaction scores improved in the latest survey.",
    "New hire orientation starts on Monday morning.",
    "Budget planning session moved to conference room B.",
    "The cafeteria menu now includes vegetarian options daily.",
    "Parking validation is available at the front desk.",
    "Holiday office closure announced for December twenty fifth.",
    "Please update your timesheet by end of week.",
    "The design review covered typography and spacing only.",
    "Sprint retrospective notes are in the shared drive.",
    "Coffee machine on third floor has been repaired.",
    "Reminder to book travel through the corporate portal.",
    "The handbook was revised for remote work policies.",
    "No action required on this automated notification.",
]

LEAK_TEMPLATES = [
    "Employee aadhaar {id12} verified for payroll.",
    "PAN {pan} submitted with tax documents.",
    "Charge card ending {cc} was declined.",
    "password=MyS3cret!2024 stored in config.",
    "api_key: sk_live_abc123xyz confidential do not share.",
    "Contact user@company.com or call +919876543210.",
    "This file is confidential internal use only.",
    "Restricted access: NDA required for {pan} holder.",
    "SSN {ssn} appears on the scanned form.",
    "AWS key AKIA{aws16} embedded in the script.",
    "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    "GitHub token gh{p} for CI pipeline.",
    "Bank transfer IFSC {ifsc} account onboarding.",
    "Database connection string password=admin123 host=db.internal",
    "Top secret missile coordinates were emailed by mistake.",
    "Private key begins with -----BEGIN RSA PRIVATE KEY-----",
    "Customer SSN {ssn2} linked to account reopening.",
    "Internal only: salary spreadsheet attached with PAN {pan}",
]


def _random_aadhaar() -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(12))


def _random_pan() -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return (
        "".join(random.choice(letters) for _ in range(5))
        + "".join(str(random.randint(0, 9)) for _ in range(4))
        + random.choice(letters)
    )


def _random_cc_shape() -> str:
    parts = ["".join(str(random.randint(0, 9)) for _ in range(4)) for _ in range(4)]
    return "-".join(parts)


def _random_ssn() -> str:
    return f"{random.randint(100, 999):03d}-{random.randint(10, 99):02d}-{random.randint(1000, 9999):04d}"


def _random_aws_tail() -> str:
    return "".join(random.choice("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(16))


def _random_ifsc() -> str:
    bank = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(4))
    tail = "0" + "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(6))
    return bank + tail


def generate_synthetic_dataset(n_per_class: int = 500) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    for _ in range(n_per_class):
        base = random.choice(SAFE_SAMPLES)
        noise = " " + " ".join(random.sample(base.split(), min(4, max(1, len(base.split())))))
        texts.append(clean_text(base + noise, remove_stopwords=False) or base)
        labels.append(0)
    for _ in range(n_per_class):
        t = random.choice(LEAK_TEMPLATES).format(
            id12=_random_aadhaar(),
            pan=_random_pan(),
            cc=_random_cc_shape(),
            ssn=_random_ssn(),
            ssn2=_random_ssn(),
            aws16=_random_aws_tail(),
            p="p_" + "x" * 36,
            ifsc=_random_ifsc(),
        )
        if random.random() < 0.35:
            t += " " + random.choice(SAFE_SAMPLES)
        texts.append(clean_text(t, remove_stopwords=False) or t)
        labels.append(1)
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


def build_pipeline(model_type: str = "random_forest") -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    if model_type == "naive_bayes":
        clf = MultinomialNB(alpha=0.1)
    elif model_type == "svm":
        clf = SGDClassifier(
            loss="modified_huber",
            penalty="l2",
            alpha=1e-4,
            max_iter=2500,
            random_state=42,
            class_weight="balanced",
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=140,
            max_depth=28,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train_and_save(
    extra_texts: Optional[List[str]] = None,
    extra_labels: Optional[List[int]] = None,
    model_type: str = "random_forest",
) -> str:
    texts, labels = generate_synthetic_dataset(520)
    if extra_texts and extra_labels and len(extra_texts) == len(extra_labels):
        texts.extend(extra_texts)
        labels.extend(extra_labels)
    pipe = build_pipeline(model_type)
    pipe.fit(texts, np.array(labels))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"pipeline": pipe, "model_type": model_type}, MODEL_PATH)
    return MODEL_PATH


def load_model():
    if not os.path.isfile(MODEL_PATH):
        train_and_save()
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle.get("model_type", "random_forest")


def predict_proba_batch(pipeline, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 2))
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(texts)
    try:
        return pipeline.predict_proba(texts)
    except Exception:
        preds = pipeline.predict(texts)
        out = np.zeros((len(texts), 2))
        out[np.arange(len(texts)), preds] = 1.0
        return out


def predict_labels(pipeline, texts: List[str], threshold: float = 0.5) -> Tuple[List[str], List[float]]:
    proba = predict_proba_batch(pipeline, texts)
    if proba.shape[1] < 2:
        p_leak = proba[:, 0]
    else:
        p_leak = proba[:, 1]
    labels = ["LEAK" if p >= threshold else "SAFE" for p in p_leak]
    return labels, list(p_leak)

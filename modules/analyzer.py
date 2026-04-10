from typing import Any, Dict, List, Optional

import pandas as pd

from modules.data_loader import dataframe_to_records, prepare_dataframe
from modules.ml_model import load_model as load_ml_bundle, predict_labels
from modules.patterns import detect_sensitive_patterns, mask_match_snippet, pattern_summary
from modules.preprocessor import clean_text, normalize_for_display, row_to_combined_text


def _build_explanation(
    regex_leak: bool,
    ml_label: str,
    ml_score: float,
    pattern_hits: str,
    final_label: str,
) -> str:
    parts: List[str] = []
    if regex_leak and pattern_hits:
        parts.append(f"Pattern match: {pattern_hits}.")
    if ml_label == "LEAK":
        parts.append(f"ML leak score {ml_score:.2f} (above threshold).")
    elif final_label == "SAFE":
        parts.append("No sensitive patterns; model score below leak threshold.")
    if final_label == "LEAK" and not regex_leak:
        parts.append("Heuristic model flagged content without a strict pattern hit.")
    return " ".join(parts) if parts else "No issues detected."


def _risk_level(regex_leak: bool, final_label: str, ml_score: float) -> str:
    if final_label == "SAFE":
        return "low"
    if regex_leak:
        return "high"
    if ml_score >= 0.75:
        return "high"
    return "medium"


def _decision_basis(regex_leak: bool, ml_label: str) -> str:
    if regex_leak and ml_label == "LEAK":
        return "regex_and_ml"
    if regex_leak:
        return "regex_only"
    if ml_label == "LEAK":
        return "ml_only"
    return "none"


def _decision_reasons(
    regex_leak: bool,
    ml_label: str,
    ml_score: float,
    ml_threshold: float,
    pattern_hits: str,
    final_label: str,
) -> List[str]:
    reasons: List[str] = []
    if regex_leak and pattern_hits:
        reasons.append(f"Sensitive pattern(s) matched: {pattern_hits}.")
    if ml_label == "LEAK":
        reasons.append(
            f"ML leak score {ml_score:.2f} is above threshold {ml_threshold:.2f}."
        )
    else:
        reasons.append(
            f"ML leak score {ml_score:.2f} is below threshold {ml_threshold:.2f}."
        )
    if final_label == "SAFE":
        reasons.append("No blocking pattern match and ML signal stayed below threshold.")
    elif regex_leak:
        reasons.append("Final verdict is LEAK because pattern detection has priority.")
    else:
        reasons.append("Final verdict is LEAK from ML confidence even without strict pattern hits.")
    return reasons


def analyze_dataframe(
    df: pd.DataFrame,
    text_columns: Optional[List[str]] = None,
    ml_threshold: float = 0.55,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    if df is None or len(df) == 0:
        _, mt = load_ml_bundle()
        return {
            "summary": {
                "total_records": 0,
                "leaks_detected": 0,
                "risk_percent": 0.0,
                "text_columns_used": [],
                "all_columns": [],
                "truncated_from": 0,
                "regex_leak_rows": 0,
                "ml_only_leak_rows": 0,
                "model_type": mt,
                "threshold_used": ml_threshold,
            },
            "rows": [],
        }

    original_len = len(df)
    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        df = df.iloc[:max_rows].copy()
    truncated_from = original_len if len(df) < original_len else 0

    df, auto_text_cols = prepare_dataframe(df)
    cols = text_columns if text_columns else auto_text_cols
    if not cols:
        cols = list(df.columns)

    pipeline, model_type = load_ml_bundle()
    records = dataframe_to_records(df)
    raw_texts: List[str] = []
    cleaned_for_ml: List[str] = []

    for rec in records:
        combined = row_to_combined_text(rec, cols)
        raw_texts.append(combined)
        cleaned = clean_text(combined, remove_stopwords=True)
        if not cleaned:
            cleaned = clean_text(combined, remove_stopwords=False) or "empty"
        cleaned_for_ml.append(cleaned)

    ml_labels, ml_scores = predict_labels(pipeline, cleaned_for_ml, threshold=ml_threshold)

    rows_out: List[Dict[str, Any]] = []
    leak_count = 0
    regex_leak_rows = 0
    ml_only_leak_rows = 0

    for i, rec in enumerate(records):
        raw = raw_texts[i]
        matches = detect_sensitive_patterns(raw)
        pattern_hits = pattern_summary(matches)
        ml_label = ml_labels[i]
        ml_score = ml_scores[i]

        regex_leak = len(matches) > 0
        final_label = "LEAK" if regex_leak or ml_label == "LEAK" else "SAFE"
        decision_basis = _decision_basis(regex_leak, ml_label)
        decision_reasons = _decision_reasons(
            regex_leak=regex_leak,
            ml_label=ml_label,
            ml_score=float(ml_score),
            ml_threshold=ml_threshold,
            pattern_hits=pattern_hits,
            final_label=final_label,
        )
        if final_label == "LEAK":
            leak_count += 1
            if regex_leak:
                regex_leak_rows += 1
            elif ml_label == "LEAK":
                ml_only_leak_rows += 1

        masked = [mask_match_snippet(m.matched_text) for m in matches]

        rows_out.append(
            {
                "row_index": i + 1,
                "preview": normalize_for_display(raw, 280),
                "ml_label": ml_label,
                "ml_leak_score": round(float(ml_score), 4),
                "regex_matches": [m.pattern_name for m in matches],
                "regex_snippets": [m.matched_text for m in matches],
                "regex_snippets_masked": masked,
                "final_label": final_label,
                "pattern_summary": pattern_hits or "-",
                "risk_level": _risk_level(regex_leak, final_label, float(ml_score)),
                "explanation": _build_explanation(
                    regex_leak, ml_label, float(ml_score), pattern_hits, final_label
                ),
                "decision_basis": decision_basis,
                "decision_reasons": decision_reasons,
                "threshold_used": ml_threshold,
            }
        )

    total = len(rows_out)
    pct = (100.0 * leak_count / total) if total else 0.0

    return {
        "summary": {
            "total_records": total,
            "leaks_detected": leak_count,
            "risk_percent": round(pct, 2),
            "text_columns_used": cols,
            "all_columns": list(df.columns),
            "truncated_from": truncated_from,
            "regex_leak_rows": regex_leak_rows,
            "ml_only_leak_rows": ml_only_leak_rows,
            "model_type": model_type,
            "threshold_used": ml_threshold,
        },
        "rows": rows_out,
    }


def analyze_single_text(text: str, ml_threshold: float = 0.55) -> Dict[str, Any]:
    df = pd.DataFrame({"text": [text]})
    return analyze_dataframe(df, text_columns=["text"], ml_threshold=ml_threshold)

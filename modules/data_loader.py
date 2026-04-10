import json
import os
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


TEXT_KINDS = {"object", "string"}
NUMERIC_TEXT_THRESHOLD = 0.3


def _decode_csv_bytes(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", engine="python")
    if ext == ".json":
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return pd.DataFrame(raw)
        if isinstance(raw, dict):
            for key in ("data", "records", "items", "rows"):
                if key in raw and isinstance(raw[key], list):
                    return pd.DataFrame(raw[key])
            return pd.json_normalize(raw)
        return pd.DataFrame([{"content": str(raw)}])
    if ext == ".txt":
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        if not lines:
            return pd.DataFrame({"text": [""]})
        return pd.DataFrame({"text": lines})
    raise ValueError(f"Unsupported format: {ext}")


def load_from_bytes(filename: str, data: bytes) -> pd.DataFrame:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".csv":
        text = _decode_csv_bytes(data)
        return pd.read_csv(StringIO(text), on_bad_lines="skip", engine="python")
    if ext == ".json":
        raw = json.loads(data.decode("utf-8", errors="replace"))
        if isinstance(raw, list):
            return pd.DataFrame(raw)
        if isinstance(raw, dict):
            for key in ("data", "records", "items", "rows"):
                if key in raw and isinstance(raw[key], list):
                    return pd.DataFrame(raw[key])
            return pd.json_normalize(raw)
        return pd.DataFrame([{"content": str(raw)}])
    if ext == ".txt":
        text = _decode_csv_bytes(data)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return pd.DataFrame({"text": [""]})
        return pd.DataFrame({"text": lines})
    raise ValueError(f"Unsupported format: {ext}")


def identify_text_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        s = df[c]
        try:
            dtype = str(s.dtype)
        except Exception:
            dtype = "object"
        if dtype in TEXT_KINDS or "object" in dtype or "string" in dtype:
            cols.append(c)
            continue
        sample = s.dropna().astype(str).head(50)
        if len(sample) == 0:
            continue
        non_numeric_ratio = sum(
            1 for v in sample if not str(v).replace(".", "", 1).isdigit()
        ) / max(len(sample), 1)
        if non_numeric_ratio >= NUMERIC_TEXT_THRESHOLD:
            cols.append(c)
    if not cols:
        cols = list(df.columns)
    return cols


def prepare_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    text_cols = identify_text_columns(df)
    for c in df.columns:
        if c in text_cols:
            df[c] = df[c].astype(str).replace({"nan": "", "None": ""})
        else:
            df[c] = df[c].fillna("")
    return df, text_cols


def dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return df.to_dict(orient="records")

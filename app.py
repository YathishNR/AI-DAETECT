import csv
import io
import json
import math
import os
import uuid
from typing import Any, Dict, List, Optional, Sequence

from flask import Blueprint, Flask, Response, current_app, jsonify, render_template, request
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import RequestEntityTooLarge

import config as app_config
from modules.analyzer import analyze_dataframe, analyze_single_text
from modules.data_loader import identify_text_columns, load_from_bytes, prepare_dataframe
from modules.ml_model import MODEL_PATH, load_model

ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".json", ".txt"}
DOWNLOAD_FORMATS = {"csv", "json"}
DEFAULT_ML_THRESHOLD = 0.55
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = int(os.environ.get("PORT", "5000"))

bp = Blueprint("api", __name__)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.after_request
def add_security_headers(response: Response):
    """
    Apply simple, low-friction security headers to every response.
    """
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("Cache-Control", "no-store")
    return response


def _json_error(message: str, status_code: int = 400, **extra: Any):
    payload: Dict[str, Any] = {"error": message}
    payload.update(extra)
    return jsonify(payload), status_code


def _server_error(message: str, exc: Exception):
    current_app.logger.exception("%s: %s", message, exc)
    if current_app.debug:
        return _json_error(f"{message}: {exc!s}", 500)
    return _json_error(message, 500)


def _extract_uploaded_file(field_name: str = "file") -> FileStorage:
    uploaded = request.files.get(field_name)
    if uploaded is None:
        raise ValueError("No file part")
    if not uploaded.filename:
        raise ValueError("No file selected")
    return uploaded


def _ensure_allowed_extension(filename: str, allowed_extensions: Sequence[str]):
    ext = os.path.splitext(filename)[1].lower()
    if ext in allowed_extensions:
        return
    readable = ", ".join(sorted(e.lstrip(".").upper() for e in allowed_extensions))
    raise ValueError(f"Unsupported format. Use {readable}.")


def _read_dataframe_from_upload(uploaded: FileStorage):
    try:
        raw = uploaded.read()
        return load_from_bytes(uploaded.filename, raw)
    except Exception as exc:
        raise ValueError(f"Could not parse file: {exc!s}") from exc


def _parse_threshold(value: Any, default: float = DEFAULT_ML_THRESHOLD) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a number between 0 and 1") from exc
    if math.isnan(parsed) or math.isinf(parsed) or parsed < 0 or parsed > 1:
        raise ValueError("threshold must be between 0 and 1")
    return parsed


def _parse_text_columns(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("text_columns must be a JSON array of column names") from exc
    if not isinstance(parsed, list):
        raise ValueError("text_columns must be a JSON array of column names")

    columns: List[str] = []
    seen = set()
    for item in parsed:
        if not isinstance(item, str):
            raise ValueError("text_columns entries must be strings")
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            columns.append(normalized)
    return columns or None


def _parse_max_rows(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_rows must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError("max_rows must be a positive integer")
    return parsed


def _serialize_list_cell(value: Any) -> Any:
    if isinstance(value, list):
        return ";".join(str(v) for v in value)
    return value


def handle_request_entity_too_large(_error):
    max_mb = int(current_app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024))
    return _json_error(f"File is too large. Maximum upload size is {max_mb} MB.", 413)


@bp.route("/api/model-info")
def model_info():
    try:
        _, model_type = load_model()
        exists = os.path.isfile(MODEL_PATH)
        return jsonify(
            {
                "model_loaded": True,
                "model_type": model_type,
                "model_path": MODEL_PATH,
                "exists_on_disk": exists,
            }
        )
    except Exception as exc:
        return _server_error("Could not load model", exc)


@bp.route("/api/preview-upload", methods=["POST"])
def preview_upload():
    try:
        uploaded = _extract_uploaded_file()
        _ensure_allowed_extension(uploaded.filename, ALLOWED_UPLOAD_EXTENSIONS)
        df = _read_dataframe_from_upload(uploaded)
    except ValueError as exc:
        return _json_error(str(exc), 400)

    try:
        df, _ = prepare_dataframe(df)
        suggested = identify_text_columns(df)
    except Exception as exc:
        return _server_error("Preview generation failed", exc)

    return jsonify(
        {
            "columns": list(df.columns),
            "suggested_text_columns": suggested,
            "row_count": len(df),
        }
    )


@bp.route("/api/analyze-upload", methods=["POST"])
def analyze_upload():
    try:
        uploaded = _extract_uploaded_file()
        _ensure_allowed_extension(uploaded.filename, ALLOWED_UPLOAD_EXTENSIONS)
        df = _read_dataframe_from_upload(uploaded)
        text_columns = _parse_text_columns(request.form.get("text_columns"))
        threshold = _parse_threshold(request.form.get("threshold"), default=DEFAULT_ML_THRESHOLD)
        max_rows = _parse_max_rows(request.form.get("max_rows"))
    except ValueError as exc:
        return _json_error(str(exc), 400)

    try:
        result = analyze_dataframe(
            df, text_columns=text_columns, ml_threshold=threshold, max_rows=max_rows
        )
    except Exception as exc:
        return _server_error("Analysis failed", exc)
    return jsonify(result)


@bp.route("/api/analyze-text", methods=["POST"])
def analyze_text():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not isinstance(text, str):
        return _json_error("text must be a string", 400)

    try:
        threshold = _parse_threshold(data.get("threshold"), default=DEFAULT_ML_THRESHOLD)
    except ValueError as exc:
        return _json_error(str(exc), 400)

    try:
        result = analyze_single_text(text, ml_threshold=threshold)
    except Exception as exc:
        return _server_error("Text analysis failed", exc)

    row = result["rows"][0] if result["rows"] else {}
    model_type = result.get("summary", {}).get("model_type")
    return jsonify(
        {
            "final_label": row.get("final_label", "SAFE"),
            "ml_label": row.get("ml_label"),
            "regex_matches": row.get("regex_matches", []),
            "regex_snippets_masked": row.get("regex_snippets_masked", []),
            "pattern_summary": row.get("pattern_summary"),
            "preview": row.get("preview"),
            "risk_level": row.get("risk_level"),
            "explanation": row.get("explanation"),
            "decision_basis": row.get("decision_basis"),
            "decision_reasons": row.get("decision_reasons", []),
            "threshold_used": row.get("threshold_used", DEFAULT_ML_THRESHOLD),
            "model_type": model_type,
        }
    )


@bp.route("/api/download-results", methods=["POST"])
def download_results():
    data = request.get_json(silent=True) or {}
    rows = data.get("rows")
    summary = data.get("summary", {})
    fmt = (data.get("format") or "csv").lower()

    if not isinstance(rows, list):
        return _json_error("rows must be an array", 400)
    if any(not isinstance(row, dict) for row in rows):
        return _json_error("each row must be an object", 400)
    if not isinstance(summary, dict):
        summary = {}
    if fmt not in DOWNLOAD_FORMATS:
        return _json_error("format must be either 'csv' or 'json'", 400)

    if fmt == "json":
        payload = {"summary": summary, "rows": rows}
        filename = f"leak_analysis_{uuid.uuid4().hex[:8]}.json"
        return Response(
            json.dumps(payload, indent=2, ensure_ascii=False),
            content_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    out = io.StringIO()
    fieldnames = [
        "row_index",
        "final_label",
        "risk_level",
        "ml_label",
        "ml_leak_score",
        "pattern_summary",
        "regex_matches",
        "regex_snippets_masked",
        "decision_basis",
        "decision_reasons",
        "threshold_used",
        "explanation",
        "preview",
    ]
    writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        normalized_row = dict(row)
        normalized_row["regex_matches"] = _serialize_list_cell(normalized_row.get("regex_matches"))
        normalized_row["regex_snippets_masked"] = _serialize_list_cell(
            normalized_row.get("regex_snippets_masked")
        )
        normalized_row["decision_reasons"] = _serialize_list_cell(
            normalized_row.get("decision_reasons")
        )
        writer.writerow(normalized_row)

    filename = f"leak_analysis_{uuid.uuid4().hex[:8]}.csv"
    return Response(
        out.getvalue(),
        content_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@bp.route("/api/health")
def health():
    return jsonify(
        {
            "model_exists": os.path.isfile(MODEL_PATH),
            "upload_folder": app_config.UPLOAD_FOLDER,
            "max_upload_mb": int(current_app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)),
        }
    )


def create_app(config_override: Optional[Dict[str, Any]] = None) -> Flask:
    """
    Application factory to support testability and environment-specific overrides.
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = app_config.MAX_CONTENT_LENGTH
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-leak-detect-key-change-me")
    app.config.setdefault("UPLOAD_FOLDER", app_config.UPLOAD_FOLDER)
    app.config.setdefault("JSON_SORT_KEYS", False)
    if config_override:
        app.config.update(config_override)

    app.register_blueprint(bp)
    app.register_error_handler(RequestEntityTooLarge, handle_request_entity_too_large)
    return app


app = create_app()


if __name__ == "__main__":
    debug_enabled = os.environ.get("FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes"}
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT, debug=debug_enabled)

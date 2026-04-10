import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PatternMatch:
    pattern_name: str
    matched_text: str


# Aadhaar: 12 digits, optional spaces
AADHAAR = re.compile(
    r"\b(?:\d{4}\s?\d{4}\s?\d{4}|\d{12})\b"
)

# Indian PAN: AAAAA9999A
PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)

# Credit card: groups of 4 digits, 13-19 total (simple Luhn-friendly shape)
CREDIT_CARD = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{1,7}\b|\b\d{13,19}\b"
)

# Password-like: password=..., pwd:..., secret key patterns
PASSWORD_HINT = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|token|auth|bearer)\s*[:=]\s*[^\s,;\"']{4,}",
)
CONFIDENTIAL_KEYWORD = re.compile(
    r"(?i)\b(confidential|classified|do not distribute|internal use only|"
    r"proprietary|restricted access|nda|non-disclosure|top secret)\b"
)

# Email (PII)
EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)

# Indian phone
PHONE_IN = re.compile(
    r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b"
)

# US SSN (dashed or spaced)
SSN_US = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b|\b\d{3}\s+\d{2}\s+\d{4}\b"
)

# US phone
PHONE_US = re.compile(
    r"\b(?:\+1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b"
)

# IPv4 (validated below)
IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b"
)

# AWS access key id
AWS_ACCESS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")

# JWT-shaped string (header.payload.signature)
JWT_LIKE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
)

# GitHub classic PAT (ghp_, gho_, ghs_, etc.)
GITHUB_PAT = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", re.IGNORECASE)

# Indian IFSC
IFSC = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")


def _luhn_valid(digits: str) -> bool:
    d = [int(c) for c in digits if c.isdigit()]
    if len(d) < 13 or len(d) > 19:
        return False
    s = 0
    alt = False
    for x in reversed(d):
        if alt:
            x = x * 2
            if x > 9:
                x -= 9
        s += x
        alt = not alt
    return s % 10 == 0


def mask_match_snippet(snippet: str, keep_edges: int = 2) -> str:
    """Redact middle of a sensitive substring for safe display."""
    s = snippet.strip()
    if len(s) <= keep_edges * 2 + 1:
        return "[redacted]"
    return s[:keep_edges] + "\u2026" + s[-keep_edges:]


PATTERN_ORDER: List[Tuple[str, re.Pattern]] = [
    ("aadhaar", AADHAAR),
    ("pan", PAN),
    ("credit_card", CREDIT_CARD),
    ("ssn_us", SSN_US),
    ("aws_access_key", AWS_ACCESS_KEY),
    ("github_token", GITHUB_PAT),
    ("jwt_like", JWT_LIKE),
    ("ifsc", IFSC),
    ("password_or_secret", PASSWORD_HINT),
    ("confidential_keyword", CONFIDENTIAL_KEYWORD),
    ("email", EMAIL),
    ("phone_in", PHONE_IN),
    ("phone_us", PHONE_US),
    ("ipv4", IPV4),
]


def detect_sensitive_patterns(text: str) -> List[PatternMatch]:
    if not text or not isinstance(text, str):
        return []
    found: List[PatternMatch] = []
    seen_spans = set()
    for name, pat in PATTERN_ORDER:
        for m in pat.finditer(text):
            span = m.span()
            if span in seen_spans:
                continue
            snippet = m.group(0)
            if name == "credit_card":
                digits_only = "".join(c for c in snippet if c.isdigit())
                if not _luhn_valid(digits_only):
                    continue
            if name == "ipv4":
                parts = snippet.split(".")
                if len(parts) != 4:
                    continue
                try:
                    if any(int(p) > 255 for p in parts):
                        continue
                except ValueError:
                    continue
            seen_spans.add(span)
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            found.append(PatternMatch(pattern_name=name, matched_text=snippet))
    return found


def has_sensitive_pattern(text: str) -> bool:
    return len(detect_sensitive_patterns(text)) > 0


def pattern_summary(matches: List[PatternMatch]) -> str:
    if not matches:
        return ""
    names = [m.pattern_name for m in matches]
    return ", ".join(sorted(set(names)))

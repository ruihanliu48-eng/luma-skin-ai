"""LumaSkin AI: a single-file Streamlit prototype for cosmetic skin analysis.

The app keeps the implementation intentionally small:
- Streamlit for the interface
- SQLite for accounts, analysis history, and a simulated product catalog
- The HKBU chat-completions endpoint for vision analysis
- Pillow for image normalization and Plotly for the KPI dashboard
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import io
import json
import math
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

APP_NAME = "LumaSkin AI"
API_ENDPOINT = "https://genai.hkbu.edu.hk/api/v1/chat/completions"
API_MODEL = "gpt-6-luna"

# Relative to the process working directory. Streamlit Community Cloud runs the
# app from the repository root, so the database is created at the project root.
DB_PATH = Path("skin_app.db")

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_SIDE = 1280
PBKDF2_ITERATIONS = 600_000
Image.MAX_IMAGE_PIXELS = 40_000_000

PAGE_OPTIONS = ("Analyze", "History & Trends", "Recommendations", "About")
CONCERN_OPTIONS = (
    "breakouts",
    "redness",
    "dryness",
    "oiliness",
    "dullness",
    "texture",
    "dark spots",
    "fine lines",
    "sensitivity",
    "pores",
)

# Fictional catalog used only to demonstrate the recommendation feature.
DEMO_PRODUCTS: list[dict[str, Any]] = [
    {
        "name": "Clear Start Gel Cleanser",
        "category": "Cleanser",
        "concerns": ["breakouts", "oiliness", "pores"],
        "price_cents": 1800,
        "description": "A low-foam gel cleanser that lifts excess oil without a tight finish.",
    },
    {
        "name": "Soft Reset Cream Cleanser",
        "category": "Cleanser",
        "concerns": ["dryness", "sensitivity", "redness"],
        "price_cents": 2200,
        "description": "A cushiony cream cleanser for a quiet, comfortable cleanse.",
    },
    {
        "name": "Daily Balance Foam",
        "category": "Cleanser",
        "concerns": ["oiliness", "pores"],
        "price_cents": 1600,
        "description": "A soft everyday foam that rinses clean without stripping.",
    },
    {
        "name": "Smooth Operator BHA Toner",
        "category": "Exfoliant",
        "concerns": ["texture", "pores", "breakouts"],
        "price_cents": 2400,
        "description": "A gentle liquid exfoliant for congested pores and uneven texture.",
    },
    {
        "name": "Dew Drop Hydrating Serum",
        "category": "Serum",
        "concerns": ["dryness", "fine lines"],
        "price_cents": 2900,
        "description": "A lightweight layer of water-binding hydration for thirsty skin.",
    },
    {
        "name": "Calm Current Serum",
        "category": "Serum",
        "concerns": ["redness", "sensitivity", "texture"],
        "price_cents": 3200,
        "description": "A soothing serum for skin that feels reactive or easily flushed.",
    },
    {
        "name": "Glow Protocol Vitamin C",
        "category": "Serum",
        "concerns": ["dullness", "dark spots", "texture"],
        "price_cents": 3600,
        "description": "A brightening antioxidant serum for a more even-looking tone.",
    },
    {
        "name": "Barrier Cloud Moisturizer",
        "category": "Moisturizer",
        "concerns": ["dryness", "sensitivity", "fine lines"],
        "price_cents": 2700,
        "description": "A rich but non-greasy cream that supports a comfortable barrier.",
    },
    {
        "name": "Matte Horizon Gel Cream",
        "category": "Moisturizer",
        "concerns": ["oiliness", "pores"],
        "price_cents": 2300,
        "description": "A weightless gel cream that softens shine and keeps skin comfortable.",
    },
    {
        "name": "Night Shift Retinol 0.3%",
        "category": "Treatment",
        "concerns": ["fine lines", "texture", "dark spots"],
        "price_cents": 3900,
        "description": "A slow-introduction night treatment for texture and early line support.",
    },
    {
        "name": "Spot Check Gel",
        "category": "Treatment",
        "concerns": ["breakouts", "oiliness"],
        "price_cents": 1400,
        "description": "A targeted gel for the occasional visible blemish.",
    },
    {
        "name": "Daily Armor SPF 50",
        "category": "Sunscreen",
        "concerns": ["dark spots", "fine lines"],
        "price_cents": 2600,
        "description": "A comfortable daily sunscreen that layers well under makeup.",
    },
]

SYSTEM_PROMPT = """You are a careful cosmetic image-assessment assistant.
Analyze only visible, non-identifying skin cues in the supplied face photo.
Do not infer identity or ethnicity. Do not provide a medical diagnosis.
Return only one valid JSON object. Never use Markdown, code fences, or commentary.
"""

USER_PROMPT = """Assess the visible skin in the attached photo.

Return exactly this JSON object and nothing else:
{
  "skin_cleanliness_kpi": 7.5,
  "skin_quality_kpi": 6.8,
  "skin_condition_description": "Two or three concise English sentences about visible texture, tone, and cleanliness.",
  "estimated_age": 29
}

Rules:
- skin_cleanliness_kpi: number from 0 to 10. Higher means less visible oil, residue, and congestion.
- skin_quality_kpi: number from 0 to 10. Higher means more even, comfortable-looking texture and tone.
- skin_condition_description: 2-3 short English sentences, observational and non-medical.
- estimated_age: integer from 1 to 120. This is a visual estimate only.
- Never return null values.
- Do not add extra keys.
- If no clear frontal face is visible, return only {"error": "short reason"}.
"""


class AppError(Exception):
    """Base class for expected, user-facing failures."""


class AuthError(AppError):
    """An account validation error."""


class DatabaseError(AppError):
    """A database initialization or access error."""


class AnalysisError(AppError):
    """An expected analysis or API failure."""

    def __init__(self, message: str, technical_detail: str | None = None) -> None:
        super().__init__(message)
        self.technical_detail = technical_detail


class LLMServiceError(AnalysisError):
    """The upstream API could not complete the request."""


class LLMResponseError(AnalysisError):
    """The upstream response could not be validated."""


class ImageValidationError(AppError):
    """The uploaded image is unusable."""


@dataclass(frozen=True)
class SkinAnalysis:
    skin_cleanliness_kpi: float
    skin_quality_kpi: float
    skin_condition_description: str
    estimated_age: int


PAGE_CSS = """
<style>
:root {
  --ink: #20352f;
  --muted: #64756e;
  --forest: #2f6b5f;
  --sage: #a9c3b7;
  --paper: #fffdf8;
  --line: #ded7ca;
  --clay: #b97856;
}
html, body, [class*="css"] {
  font-family: "Segoe UI", "Avenir Next", "Helvetica Neue", sans-serif;
}
.stApp {
  background:
    radial-gradient(circle at 8% -4%, rgba(169, 195, 183, 0.55), transparent 31rem),
    radial-gradient(circle at 100% 20%, rgba(226, 198, 169, 0.28), transparent 28rem),
    linear-gradient(180deg, #faf7f0 0%, #f4efe6 100%);
  color: var(--ink);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 2.1rem; padding-bottom: 4.5rem; }
[data-testid="stSidebar"] {
  background: #eee8dd;
  border-right: 1px solid #ded5c7;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.8rem; }
h1, h2, h3 {
  font-family: Georgia, "Times New Roman", serif !important;
  color: var(--ink);
  letter-spacing: -0.025em;
}
p, label, .stMarkdown { color: var(--ink); }
[data-testid="stCaptionContainer"] { color: var(--muted); }
.eyebrow {
  color: var(--forest);
  text-transform: uppercase;
  letter-spacing: 0.18em;
  font-size: 0.68rem;
  font-weight: 700;
}
.page-head { margin-bottom: 1.5rem; }
.page-head h1 { font-size: clamp(2rem, 4vw, 3.3rem); line-height: 1.03; margin: 0.25rem 0 0.55rem; }
.page-head p { color: var(--muted); max-width: 720px; line-height: 1.7; margin: 0; }
.section-title { margin: 0.25rem 0 0.2rem; font-size: 1.25rem; }
.section-note { color: var(--muted); font-size: 0.9rem; margin-bottom: 0.9rem; }
.side-brand { display: flex; gap: 0.7rem; align-items: center; margin-bottom: 1rem; }
.mark {
  width: 42px; height: 42px; border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  color: #fffdf7; background: var(--forest);
  font-family: Georgia, serif; font-size: 0.9rem; letter-spacing: 0.02em;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.25);
}
.side-name { font-family: Georgia, serif; font-size: 1.18rem; line-height: 1.05; color: var(--ink); }
.side-sub { font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
.side-user {
  background: rgba(255,253,248,0.6);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 0.75rem 0.9rem;
  font-size: 0.86rem;
  color: var(--muted);
  margin-bottom: 0.7rem;
}
.side-user strong { color: var(--ink); font-size: 0.95rem; }
.side-foot { color: var(--muted); font-size: 0.72rem; line-height: 1.55; }

.auth-brand { display: flex; gap: 0.75rem; align-items: center; margin-bottom: 1.6rem; }
.auth-title {
  font-family: Georgia, serif;
  font-size: clamp(2.8rem, 6vw, 4.7rem);
  line-height: 0.98;
  margin: 0.5rem 0 1rem;
  max-width: 620px;
}
.auth-lede { color: var(--muted); font-size: 1.02rem; line-height: 1.8; max-width: 540px; }
.auth-points { margin-top: 1.4rem; display: grid; gap: 0.6rem; max-width: 470px; }
.auth-point { display: flex; gap: 0.65rem; align-items: flex-start; color: #3d4f48; font-size: 0.9rem; }
.auth-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--clay); margin-top: 0.42rem; flex: 0 0 auto; }
.auth-foot { color: var(--muted); font-size: 0.75rem; line-height: 1.6; margin-top: 1rem; }

.metric-card {
  background: rgba(255,253,248,0.94);
  border: 1px solid var(--line);
  border-top: 3px solid var(--accent, var(--forest));
  border-radius: 18px;
  padding: 1rem 1.05rem 0.9rem;
  min-height: 156px;
  box-shadow: 0 10px 28px rgba(62,73,63,0.055);
}
.metric-label { color: var(--muted); font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700; }
.metric-value { font-family: Georgia, serif; font-size: 2.45rem; line-height: 1.1; margin: 0.35rem 0 0.25rem; }
.metric-unit { font-family: "Segoe UI", sans-serif; font-size: 0.85rem; color: var(--muted); margin-left: 0.15rem; }
.meter { height: 6px; border-radius: 999px; background: #eae5da; overflow: hidden; margin-top: 0.55rem; }
.meter span { display: block; height: 100%; border-radius: inherit; background: var(--accent, var(--forest)); }
.metric-note { color: var(--muted); font-size: 0.74rem; margin-top: 0.55rem; line-height: 1.4; }

.age-card {
  display: flex; justify-content: space-between; align-items: center;
  background: rgba(255,253,248,0.9); border: 1px solid var(--line);
  border-radius: 16px; padding: 0.8rem 1rem; margin-top: 0.7rem;
}
.age-label { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; }
.age-value { font-family: Georgia, serif; font-size: 1.6rem; color: var(--ink); }
.age-value span { font-family: "Segoe UI", sans-serif; font-size: 0.78rem; color: var(--muted); margin-left: 0.2rem; }

.insight-panel {
  background: rgba(255,253,248,0.92);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1rem 1.1rem;
  margin: 0.8rem 0 0.8rem;
}
.insight-panel p { font-size: 0.98rem; line-height: 1.7; margin: 0.35rem 0 0.8rem; }
.pill-row { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.pill {
  display: inline-block; padding: 0.2rem 0.58rem; border-radius: 999px;
  background: #edf2ed; color: #36564b; border: 1px solid #d5e1d8;
  font-size: 0.7rem; font-weight: 600;
}
.callout {
  border-left: 3px solid var(--forest);
  background: rgba(241,246,241,0.72);
  border-radius: 0 14px 14px 0;
  padding: 0.8rem 1rem; color: #426055; font-size: 0.86rem; line-height: 1.6;
  margin: 0.8rem 0;
}
.callout.warm { border-left-color: var(--clay); background: rgba(250,243,235,0.82); color: #785b48; }

.stat-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.75rem; margin: 0.5rem 0 1rem; }
.stat-card { background: rgba(255,253,248,0.86); border: 1px solid var(--line); border-radius: 16px; padding: 0.9rem 1rem; }
.stat-label { color: var(--muted); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; }
.stat-value { font-family: Georgia, serif; font-size: 1.75rem; line-height: 1.15; margin: 0.28rem 0 0.15rem; }
.stat-help { color: var(--muted); font-size: 0.7rem; }

.empty-card {
  background: rgba(255,253,248,0.78);
  border: 1px dashed #cfc4b3;
  border-radius: 20px;
  padding: 2rem 1.4rem;
  text-align: center;
  margin: 0.8rem 0;
}
.empty-card h3 { margin: 0.4rem 0; }
.empty-card p { color: var(--muted); max-width: 430px; margin: 0 auto 0.9rem; line-height: 1.6; }

.product-card {
  background: rgba(255,253,248,0.94);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1rem 1.05rem;
  min-height: 285px;
  display: flex;
  flex-direction: column;
  box-shadow: 0 10px 28px rgba(62,73,63,0.045);
}
.product-top { display: flex; justify-content: space-between; gap: 0.7rem; align-items: flex-start; }
.product-cat { color: var(--forest); font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 700; }
.product-price { font-family: Georgia, serif; font-size: 1.2rem; white-space: nowrap; }
.product-name { font-family: Georgia, serif; font-size: 1.2rem; line-height: 1.2; margin: 0.55rem 0 0.35rem; }
.product-desc { color: var(--muted); font-size: 0.82rem; line-height: 1.5; margin: 0 0 0.65rem; }
.product-concerns { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-bottom: 0.6rem; }
.product-why { margin-top: auto; padding-top: 0.65rem; border-top: 1px solid #ece5d9; color: #3f5d52; font-size: 0.76rem; line-height: 1.5; }

.about-card {
  background: rgba(255,253,248,0.9);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1.1rem 1.2rem;
  min-height: 165px;
  margin-bottom: 0.8rem;
}
.about-card h3 { margin: 0.3rem 0 0.45rem; font-size: 1.15rem; }
.about-card p, .about-card li { color: #4a5b54; font-size: 0.85rem; line-height: 1.65; }
.about-card ul { padding-left: 1.1rem; margin: 0.4rem 0; }

[data-testid="stForm"] {
  border: 0 !important;
  background: transparent !important;
  padding: 0 !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stForm"]) {
  background: rgba(255,253,248,0.92);
  border: 1px solid rgba(222,215,202,0.95) !important;
  border-radius: 24px;
  padding: 0.55rem 0.35rem;
  box-shadow: 0 24px 55px rgba(61,72,62,0.09);
}
[data-testid="stAppDeployButton"], [data-testid="stToolbar"] { display: none !important; }
[data-testid="stHeaderActionElements"], h1 a, h2 a, h3 a, h4 a { display: none !important; }
[data-baseweb="input"] > div,
[data-baseweb="select"] > div,
[data-baseweb="textarea"] {
  background-color: #fffdf8 !important;
  border-color: #d9d1c4 !important;
}
.stButton > button, .stFormSubmitButton > button {
  border-radius: 999px;
  border: 1px solid #b9c9bf;
  font-weight: 600;
  min-height: 2.5rem;
}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--forest) !important;
  border-color: var(--forest) !important;
  color: #fffdf8 !important;
}
[data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-primaryFormSubmit"] p {
  color: #fffdf8 !important;
}
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {
  background: #24564b !important;
  border-color: #24564b !important;
}
[data-testid="stBaseButton-primary"]:disabled,
[data-testid="stBaseButton-primaryFormSubmit"]:disabled,
.stButton > button:disabled {
  opacity: 0.42 !important;
  cursor: not-allowed !important;
}
[data-testid="stFileUploaderDropzone"] {
  border-radius: 16px;
  background: rgba(255,253,248,0.72);
  border: 1px dashed #bcae9b;
}
[data-testid="stExpander"] {
  background: rgba(255,253,248,0.58);
  border: 1px solid var(--line);
  border-radius: 14px;
}
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
[data-testid="stImage"] img { max-width: 100%; height: auto; border-radius: 14px; }
[data-testid="stTabs"] button { font-weight: 600; }

@media (max-width: 860px) {
  .stat-grid { grid-template-columns: 1fr 1fr; }
  .block-container { padding-top: 1.2rem; }
  .auth-title { font-size: 2.9rem; }
}
@media (max-width: 540px) {
  .stat-grid { grid-template-columns: 1fr; }
  .page-head h1 { font-size: 2rem; }
}
</style>
"""


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


@contextmanager
def db_connection() -> Iterator[sqlite3.Connection]:
    """Open a short-lived connection so Streamlit reruns remain thread-safe."""

    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
    except sqlite3.Error as exc:
        raise DatabaseError("Could not open the local database.") from exc

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables, indexes, and the simulated catalog if needed."""

    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with db_connection() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    skin_cleanliness_kpi REAL NOT NULL
                        CHECK (skin_cleanliness_kpi >= 0 AND skin_cleanliness_kpi <= 10),
                    skin_quality_kpi REAL NOT NULL
                        CHECK (skin_quality_kpi >= 0 AND skin_quality_kpi <= 10),
                    skin_condition_description TEXT NOT NULL,
                    estimated_age INTEGER NOT NULL
                        CHECK (estimated_age >= 1 AND estimated_age <= 120),
                    model TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_analyses_user_created
                    ON analyses(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL,
                    concerns TEXT NOT NULL,
                    price_cents INTEGER NOT NULL CHECK (price_cents >= 0),
                    description TEXT NOT NULL
                );
                """
            )
            _seed_products(conn)
    except DatabaseError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise DatabaseError(
            "Could not initialize the local database. Check the folder permissions."
        ) from exc


def _seed_products(conn: sqlite3.Connection) -> None:
    """Keep the simulated catalog in sync without creating duplicate rows."""

    for product in DEMO_PRODUCTS:
        conn.execute(
            """
            INSERT INTO products (name, category, concerns, price_cents, description)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                category = excluded.category,
                concerns = excluded.concerns,
                price_cents = excluded.price_cents,
                description = excluded.description
            """,
            (
                product["name"],
                product["category"],
                json.dumps(product["concerns"]),
                int(product["price_cents"]),
                product["description"],
            ),
        )


def fetch_products() -> list[dict[str, Any]]:
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT name, category, concerns, price_cents, description FROM products ORDER BY category, name"
        ).fetchall()

    products: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["concerns"] = json.loads(item["concerns"])
        except (TypeError, json.JSONDecodeError):
            item["concerns"] = []
        products.append(item)
    return products


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def validate_username(username: str) -> str:
    cleaned = username.strip()
    if not 3 <= len(cleaned) <= 24:
        raise AuthError("Username must be 3-24 characters long.")
    if not re.fullmatch(r"[A-Za-z0-9_]+", cleaned):
        raise AuthError("Username can use letters, numbers, and underscores only.")
    return cleaned


def validate_password(password: str) -> None:
    if not 8 <= len(password) <= 128:
        raise AuthError("Password must be 8-128 characters long.")


def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(
        iterations,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations_text, salt_text, digest_text = stored.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_text)
        if not 100_000 <= iterations <= 2_000_000:
            return False
        salt = base64.b64decode(salt_text, validate=True)
        expected = base64.b64decode(digest_text, validate=True)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def create_user(username: str, password: str) -> dict[str, Any]:
    clean_username = validate_username(username)
    validate_password(password)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with db_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (clean_username, hash_password(password), now),
            )
            user_id = int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise AuthError("That username is already taken. Choose another one.") from exc
    except sqlite3.Error as exc:
        raise DatabaseError("Could not create the account.") from exc
    return {"id": user_id, "username": clean_username, "created_at": now}


def authenticate(username: str, password: str) -> dict[str, Any] | None:
    clean_username = username.strip()
    try:
        with db_connection() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
                (clean_username,),
            ).fetchone()
    except sqlite3.Error as exc:
        raise DatabaseError("Could not read the account database.") from exc

    if row is None:
        # Keep missing-user and wrong-password work roughly comparable.
        hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), b"lumaskin-dummy-salt", PBKDF2_ITERATIONS
        )
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    return {"id": row["id"], "username": row["username"], "created_at": row["created_at"]}


def get_user(user_id: int) -> dict[str, Any] | None:
    try:
        with db_connection() as conn:
            row = conn.execute(
                "SELECT id, username, created_at FROM users WHERE id = ?", (user_id,)
            ).fetchone()
    except sqlite3.Error as exc:
        raise DatabaseError("Could not read the account database.") from exc
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Analysis history
# ---------------------------------------------------------------------------


def save_analysis(user_id: int, result: SkinAnalysis) -> tuple[int, str]:
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO analyses
                    (user_id, created_at, skin_cleanliness_kpi, skin_quality_kpi,
                     skin_condition_description, estimated_age, model)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    created_at,
                    result.skin_cleanliness_kpi,
                    result.skin_quality_kpi,
                    result.skin_condition_description,
                    result.estimated_age,
                    API_MODEL,
                ),
            )
            analysis_id = int(cursor.lastrowid)
    except sqlite3.Error as exc:
        raise DatabaseError("The analysis could not be saved to your history.") from exc
    return analysis_id, created_at


def list_analyses(user_id: int, limit: int = 500) -> list[dict[str, Any]]:
    try:
        with db_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, skin_cleanliness_kpi, skin_quality_kpi,
                       skin_condition_description, estimated_age, model
                FROM analyses
                WHERE user_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
    except sqlite3.Error as exc:
        raise DatabaseError("Could not read your analysis history.") from exc
    return [dict(row) for row in rows]


def latest_analysis(user_id: int) -> dict[str, Any] | None:
    rows = list_analyses(user_id, limit=1)
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------


def prepare_image(raw_bytes: bytes) -> bytes:
    """Validate, orient, flatten, resize, and re-encode an uploaded image as JPEG."""

    if not raw_bytes:
        raise ImageValidationError("The uploaded file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise ImageValidationError("The image is larger than 8 MB. Upload a smaller file.")

    try:
        with Image.open(io.BytesIO(raw_bytes)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(raw_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            image.load()
            width, height = image.size
            if min(width, height) < 200:
                raise ImageValidationError(
                    "The image is too small. Use a photo at least 200 x 200 pixels."
                )
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)
            rgb = background.convert("RGB")
            rgb.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            rgb.save(output, format="JPEG", quality=88, optimize=True)
            return output.getvalue()
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        raise ImageValidationError(
            "The file is not a readable JPG, PNG, or WebP image."
        ) from exc


# ---------------------------------------------------------------------------
# LLM request and strict parsing
# ---------------------------------------------------------------------------


def _extract_message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseError(
            "The API response did not contain any choices.",
            technical_detail=json.dumps(payload, ensure_ascii=False)[:3000],
        )

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise LLMResponseError(
            "The API response did not contain an assistant message.",
            technical_detail=json.dumps(payload, ensure_ascii=False)[:3000],
        )

    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        text = "\n".join(parts)
    else:
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            raise AnalysisError(f"The model declined the request: {refusal[:300]}")
        text = ""

    if not text.strip():
        raise LLMResponseError(
            "The API returned an empty analysis.",
            technical_detail=json.dumps(payload, ensure_ascii=False)[:3000],
        )
    return text.strip()


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LLMResponseError(f"Field '{field}' must be a number.")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 10:
        raise LLMResponseError(f"Field '{field}' must be between 0 and 10.")
    return number


def _load_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise LLMResponseError(
                "The model did not return valid JSON. Please retry the analysis.",
                technical_detail=text[:4000],
            ) from None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                "The model returned malformed JSON. Please retry the analysis.",
                technical_detail=text[:4000],
            ) from exc

    if not isinstance(value, dict):
        raise LLMResponseError(
            "The model response must be a JSON object.",
            technical_detail=text[:4000],
        )
    return value


def parse_analysis(text: str) -> SkinAnalysis:
    """Parse and strictly validate the four required model fields."""

    payload = _load_json_object(text)

    if "error" in payload and len(payload) == 1:
        message = str(payload["error"]).strip()[:400]
        raise AnalysisError(
            f"The image could not be assessed: {message or 'no clear face was detected.'}",
            technical_detail=text[:4000],
        )

    required = {
        "skin_cleanliness_kpi",
        "skin_quality_kpi",
        "skin_condition_description",
        "estimated_age",
    }
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        detail_parts = []
        if missing:
            detail_parts.append("missing: " + ", ".join(missing))
        if extra:
            detail_parts.append("unexpected: " + ", ".join(extra))
        raise LLMResponseError(
            "The model response did not match the required JSON schema"
            + (f" ({'; '.join(detail_parts)})." if detail_parts else "."),
            technical_detail=text[:4000],
        )

    cleanliness = _number(payload["skin_cleanliness_kpi"], "skin_cleanliness_kpi")
    quality = _number(payload["skin_quality_kpi"], "skin_quality_kpi")

    description = payload["skin_condition_description"]
    if not isinstance(description, str) or not description.strip():
        raise LLMResponseError(
            "Field 'skin_condition_description' must be non-empty text."
        )
    description = description.strip()
    if len(description) > 1000:
        raise LLMResponseError("Field 'skin_condition_description' is too long.")

    age = payload["estimated_age"]
    if isinstance(age, bool) or not isinstance(age, int) or not 1 <= age <= 120:
        raise LLMResponseError(
            "Field 'estimated_age' must be a whole number from 1 to 120."
        )

    return SkinAnalysis(
        skin_cleanliness_kpi=round(cleanliness, 1),
        skin_quality_kpi=round(quality, 1),
        skin_condition_description=description,
        estimated_age=age,
    )


def analyze_face(image_jpeg: bytes, api_key: str) -> SkinAnalysis:
    """Send a normalized JPEG to the chat-completions endpoint."""

    if not api_key.strip():
        raise AnalysisError("Enter an API key in the sidebar before analyzing.")

    encoded = base64.b64encode(image_jpeg).decode("ascii")
    payload = {
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    },
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": 500,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    for attempt in range(2):
        try:
            response = requests.post(
                API_ENDPOINT, headers=headers, json=payload, timeout=(10, 120)
            )
        except requests.Timeout as exc:
            raise LLMServiceError(
                "The request timed out. Check your connection and try again.",
                technical_detail=str(exc),
            ) from exc
        except requests.RequestException as exc:
            raise LLMServiceError(
                "Could not reach the HKBU API. Check your internet connection.",
                technical_detail=str(exc),
            ) from exc

        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 0:
                time.sleep(1.2)
                continue
            raise LLMServiceError(
                "The API is temporarily unavailable. Wait a moment and retry.",
                technical_detail=f"HTTP {response.status_code}: {response.text[:2000]}",
            )

        if response.status_code in (401, 403):
            raise LLMServiceError(
                "The API key was rejected. Check the key and try again.",
                technical_detail=f"HTTP {response.status_code}: {response.text[:2000]}",
            )
        if response.status_code == 404:
            raise LLMServiceError(
                "The API endpoint was not found. Check the configured endpoint.",
                technical_detail=f"HTTP 404: {response.text[:2000]}",
            )
        if response.status_code != 200:
            raise LLMServiceError(
                f"The API rejected the request (HTTP {response.status_code}).",
                technical_detail=f"HTTP {response.status_code}: {response.text[:2000]}",
            )

        try:
            response_payload = response.json()
        except ValueError as exc:
            raise LLMResponseError(
                "The API returned a non-JSON response.",
                technical_detail=response.text[:4000],
            ) from exc
        if not isinstance(response_payload, dict):
            raise LLMResponseError(
                "The API returned an unexpected response shape.",
                technical_detail=response.text[:4000],
            )
        if response_payload.get("error"):
            raise LLMServiceError(
                "The API could not complete the analysis.",
                technical_detail=json.dumps(response_payload, ensure_ascii=False)[:3000],
            )

        content = _extract_message_text(response_payload)
        try:
            return parse_analysis(content)
        except LLMResponseError as exc:
            if not exc.technical_detail:
                exc.technical_detail = content[:4000]
            raise

    raise LLMServiceError("The analysis could not be completed. Please retry.")


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def _score_product(
    product: dict[str, Any],
    focus: str,
    concerns: list[str],
    cleanliness: float,
    quality: float,
) -> float:
    category = product["category"]
    score = 0.0

    if focus == "cleanliness":
        score += {
            "Cleanser": 5.0,
            "Exfoliant": 4.0,
            "Treatment": 3.0,
            "Moisturizer": 2.0,
            "Sunscreen": 2.0,
            "Serum": 1.0,
        }.get(category, 0.0)
        if cleanliness < 5 and category in {"Cleanser", "Exfoliant", "Treatment"}:
            score += 3.0
    elif focus == "quality":
        score += {
            "Serum": 5.0,
            "Moisturizer": 5.0,
            "Treatment": 2.0,
            "Cleanser": 2.0,
            "Sunscreen": 2.0,
            "Exfoliant": 1.0,
        }.get(category, 0.0)
        if quality < 5 and category in {"Serum", "Moisturizer"}:
            score += 3.0
    else:
        score += {
            "Sunscreen": 5.0,
            "Moisturizer": 4.0,
            "Cleanser": 3.0,
            "Serum": 2.0,
            "Treatment": 1.0,
            "Exfoliant": 1.0,
        }.get(category, 0.0)

    overlap = set(concerns) & set(product["concerns"])
    score += 4.0 * len(overlap)

    if "redness" in concerns and "redness" in product["concerns"]:
        score += 1.0
    return score


def recommend_products(
    cleanliness: float,
    quality: float,
    concerns: list[str],
) -> list[dict[str, Any]]:
    """Return 1-3 catalog products driven by the lowest KPI and user concerns."""

    products = fetch_products()
    if not products:
        return []

    if abs(cleanliness - quality) < 0.5:
        focus = "balance"
    elif cleanliness < quality:
        focus = "cleanliness"
    else:
        focus = "quality"

    ranked = sorted(
        products,
        key=lambda product: _score_product(
            product, focus, concerns, cleanliness, quality
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    used_categories: set[str] = set()
    for product in ranked:
        if product["category"] in used_categories:
            continue
        used_categories.add(product["category"])
        selected.append(product)
        if len(selected) == 3:
            break

    if not selected:
        selected = products[:1]

    results: list[dict[str, Any]] = []
    for product in selected:
        overlap = [item for item in concerns if item in product["concerns"]]
        reasons: list[str] = []
        if overlap:
            reasons.append("Matches your focus on " + ", ".join(overlap[:2]) + ".")
        if focus == "cleanliness" and product["category"] in {
            "Cleanser", "Exfoliant", "Treatment"
        }:
            reasons.append("Supports the lower cleanliness KPI.")
        elif focus == "quality" and product["category"] in {"Serum", "Moisturizer"}:
            reasons.append("Supports the lower quality KPI.")
        elif focus == "balance" and product["category"] == "Sunscreen":
            reasons.append("A simple daily protection step for maintenance.")
        if not reasons:
            reasons.append("A supportive step that fits the current KPI profile.")
        results.append({"product": product, "reason": " ".join(reasons)})

    return results


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def inject_css() -> None:
    st.markdown(PAGE_CSS, unsafe_allow_html=True)


def format_utc(iso_text: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    except (ValueError, TypeError):
        return iso_text


def score_band(score: float) -> str:
    if score >= 8:
        return "Strong"
    if score >= 6.5:
        return "Balanced"
    if score >= 5:
        return "Fair"
    return "Needs care"


def page_head(eyebrow: str, title: str, description: str) -> None:
    st.html(
        f"""
        <div class="page-head">
          <div class="eyebrow">{escape(eyebrow)}</div>
          <h1>{escape(title)}</h1>
          <p>{escape(description)}</p>
        </div>
        """
    )


def metric_card(
    label: str,
    value: str,
    unit: str,
    note: str,
    accent: str,
    percent: float,
) -> str:
    safe_percent = max(0.0, min(100.0, percent))
    return f"""
    <div class="metric-card" style="--accent:{accent};">
      <div class="metric-label">{escape(label)}</div>
      <div class="metric-value">{escape(value)}<span class="metric-unit">{escape(unit)}</span></div>
      <div class="meter"><span style="width:{safe_percent:.1f}%"></span></div>
      <div class="metric-note">{escape(note)}</div>
    </div>
    """


def stat_card(label: str, value: str, helper: str) -> str:
    return f"""
    <div class="stat-card">
      <div class="stat-label">{escape(label)}</div>
      <div class="stat-value">{escape(value)}</div>
      <div class="stat-help">{escape(helper)}</div>
    </div>
    """


def _logout_callback() -> None:
    for key in (
        "auth_user_id",
        "auth_username",
        "last_result",
        "last_image_jpeg",
        "upload_digest",
        "save_warning",
        "analysis_error",
    ):
        st.session_state.pop(key, None)
    st.session_state["_clear_api_key"] = True
    st.session_state["_clear_face_upload"] = True
    st.session_state["_reset_nav"] = True


def _go_to(page: str) -> None:
    if page in PAGE_OPTIONS:
        st.session_state["nav"] = page


def current_user() -> dict[str, Any] | None:
    user_id = st.session_state.get("auth_user_id")
    if not user_id:
        return None
    user = get_user(int(user_id))
    if user is None:
        for key in ("auth_user_id", "auth_username", "last_result", "upload_digest"):
            st.session_state.pop(key, None)
    return user


# ---------------------------------------------------------------------------
# Authentication page
# ---------------------------------------------------------------------------


def render_auth_page() -> None:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.html(
            f"""
            <div class="auth-brand">
              <span class="mark">LS</span>
              <div>
                <div class="side-name">{escape(APP_NAME)}</div>
                <div class="side-sub">AI skin journal</div>
              </div>
            </div>
            <div class="eyebrow">Cosmetic insight, kept private</div>
            <div class="auth-title">Skincare,<br>understood.</div>
            <p class="auth-lede">
              Turn one clear face photo into two visible-skin KPIs, a short
              assessment, and a routine matched to your profile.
            </p>
            <div class="auth-points">
              <div class="auth-point"><span class="auth-dot"></span><span>Your analyses are isolated to your account.</span></div>
              <div class="auth-point"><span class="auth-dot"></span><span>Uploaded photos are never saved by this app.</span></div>
              <div class="auth-point"><span class="auth-dot"></span><span>See score trends over time in your private history.</span></div>
            </div>
            """
        )

    with right:
        with st.container(border=True):
            st.html("<h3>Welcome back</h3>")
            st.caption("Sign in or create an account to continue.")
            sign_in_tab, sign_up_tab = st.tabs(["Sign in", "Create account"])

            with sign_in_tab:
                with st.form("sign_in_form"):
                    login_username = st.text_input("Username", key="login_username")
                    login_password = st.text_input(
                        "Password", type="password", key="login_password"
                    )
                    submitted = st.form_submit_button(
                        "Sign in", type="primary", width="stretch"
                    )
                if submitted:
                    try:
                        user = authenticate(login_username, login_password)
                    except DatabaseError as exc:
                        st.error(str(exc))
                    else:
                        if user is None:
                            st.error("Incorrect username or password.")
                        else:
                            st.session_state["auth_user_id"] = user["id"]
                            st.session_state["auth_username"] = user["username"]
                            st.rerun()

            with sign_up_tab:
                with st.form("sign_up_form"):
                    new_username = st.text_input("Username", key="register_username")
                    new_password = st.text_input(
                        "Password",
                        type="password",
                        key="register_password",
                        help="Use 8-128 characters.",
                    )
                    confirm_password = st.text_input(
                        "Confirm password", type="password", key="register_confirm"
                    )
                    registered = st.form_submit_button(
                        "Create account", type="primary", width="stretch"
                    )
                if registered:
                    if new_password != confirm_password:
                        st.error("The passwords do not match.")
                    else:
                        try:
                            user = create_user(new_username, new_password)
                        except AuthError as exc:
                            st.error(str(exc))
                        except DatabaseError as exc:
                            st.error(str(exc))
                        else:
                            st.session_state["auth_user_id"] = user["id"]
                            st.session_state["auth_username"] = user["username"]
                            st.rerun()

            st.html(
                '<div class="auth-foot">No email is required. Authentication uses a salted '
                "PBKDF2 password hash. Sessions last until you sign out or close the tab.</div>"
            )


# ---------------------------------------------------------------------------
# Analyze page
# ---------------------------------------------------------------------------


def render_result_panel(result: dict[str, Any]) -> None:
    cleanliness = float(result["skin_cleanliness_kpi"])
    quality = float(result["skin_quality_kpi"])
    age = int(result["estimated_age"])

    st.html('<div class="eyebrow">Analysis result</div>')
    cards = st.columns(2)
    with cards[0]:
        st.html(
            metric_card(
                "Skin Cleanliness KPI",
                f"{cleanliness:.1f}",
                "/ 10",
                f"{score_band(cleanliness)} · visible residue and congestion",
                "#2f6b5f",
                cleanliness * 10,
            )
        )
    with cards[1]:
        st.html(
            metric_card(
                "Skin Quality KPI",
                f"{quality:.1f}",
                "/ 10",
                f"{score_band(quality)} · texture, tone, and comfort",
                "#b97856",
                quality * 10,
            )
        )

    st.html(
        f"""
        <div class="age-card">
          <div>
            <div class="age-label">Estimated age</div>
            <div class="age-value">{age}<span>years</span></div>
          </div>
          <div class="stat-help">Visual estimate only · not a verification</div>
        </div>
        """
    )

    st.html(
        f"""
        <div class="insight-panel">
          <div class="eyebrow">What the model observed</div>
          <p>{escape(result["skin_condition_description"]).replace(chr(10), "<br>")}</p>
        </div>
        """
    )

    if result.get("save_warning"):
        st.warning(result["save_warning"])

    action_left, action_right = st.columns(2)
    with action_left:
        st.button(
            "Open history",
            width="stretch",
            on_click=_go_to,
            args=("History & Trends",),
        )
    with action_right:
        st.button(
            "View recommendations",
            width="stretch",
            on_click=_go_to,
            args=("Recommendations",),
        )

    saved_label = (
        f"Saved {format_utc(result['created_at'])}."
        if result.get("analysis_id")
        else "Result not saved to history."
    )
    st.caption(f"{saved_label} Cosmetic assessment only.")


def render_analyze_page(user: dict[str, Any]) -> None:
    page_head(
        "A private skin workspace",
        "Analyze",
        "Upload a clear face photo to measure visible cleanliness and quality. "
        "The photo is processed in memory and is never stored.",
    )

    last = st.session_state.get("last_result")
    api_key = str(st.session_state.get("api_key", "")).strip()

    left, right = st.columns([1.0, 1.0], gap="large")

    with left:
        st.html('<h3 class="section-title">Upload a photo</h3>')
        uploaded = st.file_uploader(
            "Face photo",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            help="JPG, PNG, or WebP. Maximum 8 MB. A clear frontal face works best.",
            key="face_upload",
        )

        if uploaded is not None:
            raw_bytes = uploaded.getvalue()
            digest = hashlib.sha256(raw_bytes).hexdigest()[:20]
            if st.session_state.get("upload_digest") != digest:
                st.session_state["upload_digest"] = digest
                for key in ("last_result", "save_warning"):
                    st.session_state.pop(key, None)
            st.image(raw_bytes, width=380)
            st.caption("Privacy: the uploaded image is not written to SQLite.")
        else:
            st.html(
                """
                <div class="callout warm">
                  <strong>Photo tips</strong><br>
                  Face the camera directly, use even daylight, and remove heavy filters.
                  Bare skin gives the most consistent result.
                </div>
                """
            )

        if not api_key:
            st.info("Enter your HKBU API key in the sidebar to enable analysis.")
        can_run = uploaded is not None and bool(api_key)
        run_clicked = st.button(
            "Analyze skin",
            type="primary",
            width="stretch",
            disabled=not can_run,
            key="run_analysis",
        )

        if run_clicked and uploaded is not None:
            try:
                with st.spinner("Reading visible skin cues..."):
                    processed = prepare_image(uploaded.getvalue())
                    result = analyze_face(processed, api_key)
            except ImageValidationError as exc:
                st.error(str(exc))
            except AnalysisError as exc:
                st.error(str(exc))
                if exc.technical_detail:
                    with st.expander("Technical details"):
                        st.code(exc.technical_detail[:4000], language="json")
            except DatabaseError as exc:
                st.error(str(exc))
            else:
                try:
                    analysis_id, created_at = save_analysis(user["id"], result)
                    save_warning = None
                except DatabaseError as exc:
                    analysis_id = None
                    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    save_warning = f"{exc} The result is shown but was not saved."

                st.session_state["last_result"] = {
                    **asdict(result),
                    "analysis_id": analysis_id,
                    "created_at": created_at,
                    "save_warning": save_warning,
                }
                st.rerun()

    with right:
        if last:
            render_result_panel(last)
        else:
            st.html(
                """
                <div class="empty-card">
                  <div class="eyebrow">Awaiting a photo</div>
                  <h3>Your result will appear here.</h3>
                  <p>Upload a face photo, add your API key in the sidebar, and run the analysis.
                  You will see two KPIs, a short observation, and an estimated age.</p>
                </div>
                """
            )
            st.html(
                """
                <div class="callout">
                  <strong>What gets stored</strong><br>
                  Only the four analysis fields and a timestamp are saved to your private
                  history. The image itself is discarded after the request.
                </div>
                """
            )


# ---------------------------------------------------------------------------
# History & Trends page
# ---------------------------------------------------------------------------


def render_history_page(user: dict[str, Any]) -> None:
    page_head(
        "Private history",
        "History & Trends",
        "Track how visible cleanliness and quality change across your saved assessments. "
        "Only this account can read these records.",
    )

    records = list_analyses(user["id"])
    if not records:
        st.html(
            """
            <div class="empty-card">
              <div class="eyebrow">Nothing measured yet</div>
              <h3>Your first snapshot starts here.</h3>
              <p>Analyze a face photo and the score history will appear on this page.</p>
            </div>
            """
        )
        st.button(
            "Analyze a photo",
            type="primary",
            on_click=_go_to,
            args=("Analyze",),
        )
        return

    df = pd.DataFrame(records)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    range_col, timezone_col = st.columns([0.7, 0.3], gap="large")
    with range_col:
        window_label = st.selectbox(
            "Time range",
            ["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
            index=3,
            key="history_window",
        )
    with timezone_col:
        st.caption("Times are shown in UTC.")

    now = pd.Timestamp.now(tz="UTC")
    windows = {
        "Last 7 days": pd.Timedelta(days=7),
        "Last 30 days": pd.Timedelta(days=30),
        "Last 90 days": pd.Timedelta(days=90),
    }
    filtered = df.copy()
    if window_label in windows:
        filtered = filtered[filtered["created_at"] >= now - windows[window_label]]
    filtered = filtered.sort_values("created_at").reset_index(drop=True)
    if filtered.empty:
        st.info("No scans fall inside this time range. Choose a wider range.")
        return

    latest = filtered.iloc[-1]
    previous = filtered.iloc[-2] if len(filtered) > 1 else None
    delta_text = "First scan"
    if previous is not None:
        delta = latest["skin_cleanliness_kpi"] - previous["skin_cleanliness_kpi"]
        delta_text = f"{delta:+.1f} vs previous"

    st.html(
        f"""
        <div class="stat-grid">
          {stat_card("Scans in range", str(len(filtered)), window_label)}
          {stat_card("Latest cleanliness", f"{latest['skin_cleanliness_kpi']:.1f} / 10", delta_text)}
          {stat_card("Latest quality", f"{latest['skin_quality_kpi']:.1f} / 10", score_band(float(latest['skin_quality_kpi'])))}
          {stat_card("Average visual age", f"{filtered['estimated_age'].mean():.0f}", "Across this range")}
        </div>
        """
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered["created_at"],
            y=filtered["skin_cleanliness_kpi"],
            mode="lines+markers",
            name="Skin Cleanliness",
            line={"color": "#2f6b5f", "width": 3},
            marker={"size": 8, "line": {"color": "#fffdf8", "width": 2}},
            hovertemplate="%{x|%d %b %Y}<br>Cleanliness %{y:.1f}/10<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=filtered["created_at"],
            y=filtered["skin_quality_kpi"],
            mode="lines+markers",
            name="Skin Quality",
            line={"color": "#b97856", "width": 3},
            marker={"size": 8, "line": {"color": "#fffdf8", "width": 2}},
            hovertemplate="%{x|%d %b %Y}<br>Quality %{y:.1f}/10<extra></extra>",
        )
    )
    fig.update_layout(
        height=380,
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,253,248,0.58)",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 0},
        xaxis={"showgrid": False, "title": None, "linecolor": "#ded7ca"},
        yaxis={
            "range": [0, 10],
            "dtick": 2,
            "title": "Score / 10",
            "gridcolor": "rgba(47,107,95,0.12)",
            "zeroline": False,
        },
        font={"family": "Segoe UI, Helvetica, sans-serif", "color": "#30433d"},
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    if len(filtered) == 1:
        st.info("One scan so far. Add another photo to reveal a trend.")
    else:
        st.caption(
            f"Latest scan: {format_utc(latest['created_at'].isoformat())} · "
            f"Cleanliness {latest['skin_cleanliness_kpi']:.1f}, "
            f"Quality {latest['skin_quality_kpi']:.1f}."
        )

    st.markdown("### Saved scans")
    table = filtered.sort_values("created_at", ascending=False).copy()
    table["Date (UTC)"] = table["created_at"].dt.strftime("%d %b %Y, %H:%M")
    table["Cleanliness"] = table["skin_cleanliness_kpi"].map(lambda value: f"{value:.1f}")
    table["Quality"] = table["skin_quality_kpi"].map(lambda value: f"{value:.1f}")
    table["Estimated age"] = table["estimated_age"]
    table["Description"] = table["skin_condition_description"]
    columns = ["Date (UTC)", "Cleanliness", "Quality", "Estimated age", "Description"]
    st.dataframe(
        table[columns],
        width="stretch",
        hide_index=True,
        column_config={
            "Date (UTC)": st.column_config.TextColumn("Date (UTC)", width="medium"),
            "Description": st.column_config.TextColumn("Description", width="large"),
        },
    )
    st.caption("Scores are cosmetic visual estimates, not clinical measurements.")


# ---------------------------------------------------------------------------
# Recommendations page
# ---------------------------------------------------------------------------


def render_recommendations_page(user: dict[str, Any]) -> None:
    page_head(
        "Personalized edit",
        "Recommendations",
        "Recommendations use the lower KPI and the concerns you select. "
        "The catalog is a prototype and does not represent real products for sale.",
    )

    recent = latest_analysis(user["id"])
    mode = "Manual KPIs"
    if recent:
        mode = st.radio(
            "Base recommendations on",
            ["Latest analysis", "Manual KPIs"],
            horizontal=True,
            key="recommendation_mode",
        )

    if mode == "Latest analysis" and recent:
        cleanliness = float(recent["skin_cleanliness_kpi"])
        quality = float(recent["skin_quality_kpi"])
        st.html(
            f"""
            <div class="callout">
              Using the scan from <strong>{escape(format_utc(recent["created_at"]))}</strong>
              · Cleanliness <strong>{cleanliness:.1f}/10</strong>
              · Quality <strong>{quality:.1f}/10</strong>.
            </div>
            """
        )
    else:
        slider_left, slider_right = st.columns(2)
        with slider_left:
            cleanliness = st.slider(
                "Skin Cleanliness KPI", 0.0, 10.0, 6.0, 0.5, key="manual_cleanliness"
            )
        with slider_right:
            quality = st.slider(
                "Skin Quality KPI", 0.0, 10.0, 6.0, 0.5, key="manual_quality"
            )

    concerns = st.multiselect(
        "Current concerns",
        list(CONCERN_OPTIONS),
        key="recommendation_concerns",
        format_func=lambda value: value.title(),
        help="Used to prioritize catalog products alongside the lower KPI.",
    )

    lower_name = "cleanliness" if cleanliness < quality else "quality"
    lower_value = min(cleanliness, quality)
    st.html(
        f"""
        <div class="callout">
          Primary focus: <strong>{escape(lower_name.title())}</strong>
          at <strong>{lower_value:.1f}/10</strong>.
          Recommendations are limited to 1-3 products so the routine stays simple.
        </div>
        """
    )

    recommendations = recommend_products(cleanliness, quality, concerns)
    if not recommendations:
        st.warning("The prototype catalog is unavailable. Re-run the app to reseed the database.")
        return

    st.markdown("### Suggested products")
    columns = st.columns(3, gap="large")
    for index, item in enumerate(recommendations):
        product = item["product"]
        with columns[index]:
            concern_pills = "".join(
                f'<span class="pill">{escape(concern)}</span>'
                for concern in product["concerns"][:3]
            )
            st.html(
                f"""
                <div class="product-card">
                  <div class="product-top">
                    <div class="product-cat">{escape(product["category"])}</div>
                    <div class="product-price">${product["price_cents"] / 100:.2f}</div>
                  </div>
                  <div class="product-name">{escape(product["name"])}</div>
                  <p class="product-desc">{escape(product["description"])}</p>
                  <div class="product-concerns">{concern_pills}</div>
                  <div class="product-why"><strong>Why it fits:</strong> {escape(item["reason"])}</div>
                </div>
                """
            )

    st.caption(
        "Prototype catalog · Prices are illustrative · No purchase links · "
        "Introduce one new product at a time and patch test first."
    )


# ---------------------------------------------------------------------------
# About page
# ---------------------------------------------------------------------------


def render_about_page() -> None:
    page_head(
        "About this prototype",
        "About",
        "A small technical demonstration of image-based cosmetic assessment, "
        "private history, and rule-based product matching.",
    )

    top_left, top_right = st.columns(2, gap="large")
    with top_left:
        st.html(
            f"""
            <div class="about-card">
              <div class="eyebrow">Analysis</div>
              <h3>One photo, four outputs</h3>
              <p>The app sends a normalized JPEG to the HKBU vision endpoint using
              <strong>{escape(API_MODEL)}</strong>. The model must return strict JSON with
              two 0-10 KPIs, a short visible-skin description, and an estimated age.</p>
            </div>
            """
        )
        st.html(
            """
            <div class="about-card">
              <div class="eyebrow">Recommendations</div>
              <h3>Rule-based, not diagnostic</h3>
              <p>The prototype catalog is fictional. Product selection is driven by the
              lower KPI and the concerns selected on the Recommendations page.</p>
            </div>
            """
        )
    with top_right:
        st.html(
            """
            <div class="about-card">
              <div class="eyebrow">Your data</div>
              <h3>Private by account</h3>
              <ul>
                <li>Usernames are unique, case-insensitive.</li>
                <li>Passwords use salted PBKDF2-HMAC-SHA256 hashing.</li>
                <li>Analyses are stored in SQLite and filtered by user ID.</li>
                <li>Uploaded images are never written to the database.</li>
                <li>The API key exists only in this Streamlit session.</li>
              </ul>
            </div>
            """
        )
        st.html(
            """
            <div class="about-card">
              <div class="eyebrow">Limitations</div>
              <h3>Cosmetic guidance only</h3>
              <p>Lighting, camera angle, makeup, and filters affect results. The output is
              not a medical diagnosis or a clinical measurement. Do not use it for treatment
              decisions.</p>
            </div>
            """
        )

    st.html(
        """
        <div class="callout">
          <strong>Stack:</strong> Streamlit · SQLite · Pandas · Plotly · Requests · Pillow.
          Data stays local in <code>skin_app.db</code> for this prototype.
        </div>
        """
    )


# ---------------------------------------------------------------------------
# Sidebar and app entry
# ---------------------------------------------------------------------------


def render_sidebar(user: dict[str, Any]) -> str:
    with st.sidebar:
        st.html(
            f"""
            <div class="side-brand">
              <span class="mark">LS</span>
              <div>
                <div class="side-name">{escape(APP_NAME)}</div>
                <div class="side-sub">AI skin journal</div>
              </div>
            </div>
            <div class="side-user">Signed in as<br><strong>{escape(user["username"])}</strong></div>
            """
        )
        page = st.radio(
            "Navigation",
            PAGE_OPTIONS,
            key="nav",
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("#### API connection")
        st.text_input(
            "HKBU API key",
            type="password",
            key="api_key",
            placeholder="Paste your API key",
            help="Used only for this Streamlit session. It is never saved or logged.",
        )
        st.caption("Required for new analyses. Session only.")
        st.caption(f"Model: {API_MODEL}")

        st.divider()
        st.button(
            "Sign out",
            width="stretch",
            on_click=_logout_callback,
            key="sign_out_button",
        )
        st.html(
            '<div class="side-foot">Cosmetic assessment only. Not a medical device or '
            "diagnosis. Photos are processed in memory and never saved.</div>"
        )
    return page


def main() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="auto",
    )
    inject_css()

    if st.session_state.pop("_clear_api_key", False):
        st.session_state.pop("api_key", None)
    if st.session_state.pop("_clear_face_upload", False):
        st.session_state.pop("face_upload", None)
    if st.session_state.pop("_reset_nav", False):
        st.session_state["nav"] = "Analyze"
    if st.session_state.get("nav") not in PAGE_OPTIONS:
        st.session_state["nav"] = "Analyze"

    try:
        if not st.session_state.get("_db_ready"):
            init_db()
            st.session_state["_db_ready"] = True
    except DatabaseError as exc:
        st.error(str(exc))
        st.stop()

    try:
        user = current_user()
    except DatabaseError as exc:
        st.error(str(exc))
        st.stop()

    if user is None:
        render_auth_page()
        return

    page = render_sidebar(user)
    try:
        if page == "Analyze":
            render_analyze_page(user)
        elif page == "History & Trends":
            render_history_page(user)
        elif page == "Recommendations":
            render_recommendations_page(user)
        elif page == "About":
            render_about_page()
    except DatabaseError as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()

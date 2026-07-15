"""Inventory data access layer.

Loads the car inventory workbook (sheet ``"cleaned dataset"``), merges the
cached LLM enrichment (``data/inventory_enriched.json``) on ``Listing_ID``,
adds a cleaned description column, and exposes search / lookup helpers used by
the agent's tools.

The inventory is loaded once and cached at module level. The source xlsx is
never modified.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from app import config

SHEET_NAME = "cleaned dataset"

# Allowed / validation bounds (mirror the enrichment script).
YEAR_MIN, YEAR_MAX = 2002, 2026
PRICE_MIN, PRICE_MAX = 10_000, 5_000_000

# Columns contributed by the enrichment cache.
_ENRICH_FIELDS = [
    "price_aed",
    "monthly_payment_aed",
    "mileage_km",
    "color",
    "body_type",
    "has_warranty",
    "spec",
]

# --- description cleaning -------------------------------------------------

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Phone numbers, incl. +971 formats: an optional +, then >=9 digits possibly
# separated by spaces / dashes / parens. Comma-grouped numbers (prices,
# mileage like "54,500" / "111,000") are intentionally NOT matched.
_PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{7,}\d")
_HASHTAG_RE = re.compile(r"#\w+")
_HANDLE_RE = re.compile(r"(?<!\w)@[\w.]+")
# Emoji + wingding/private-use glyphs (e.g. ) + misc symbol blocks.
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # emoji & symbols
    "\U00002600-\U000027BF"  # misc symbols & dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000E000-\U0000F8FF"  # private use area (wingding arrows etc.)
    "\U00002190-\U000021FF"  # arrows
    "\U00002B00-\U00002BFF"  # misc symbols and arrows
    "\U00002000-\U0000206F"  # general punctuation (bullets ▪ • …)
    "\U000025A0-\U000025FF"  # geometric shapes
    "]",
    flags=re.UNICODE,
)
# Separator spam: runs of repeated separator chars (----- or ••••• etc.).
_SEPARATOR_RE = re.compile(r"[-–—•▪●○*=_~.]{3,}")
_WS_RE = re.compile(r"\s+")


def _clean_description(text: Any) -> str:
    """Strip contact info, links, emojis and separator spam from a listing.

    Removes phone numbers (incl. +971 formats), URLs, emails, hashtags,
    social handles and emojis, collapses separator spam and whitespace. The
    original ``description`` column is left untouched.
    """
    if not isinstance(text, str):
        return ""
    s = text
    s = _URL_RE.sub(" ", s)
    s = _EMAIL_RE.sub(" ", s)
    s = _PHONE_RE.sub(" ", s)
    s = _HASHTAG_RE.sub(" ", s)
    s = _HANDLE_RE.sub(" ", s)
    s = _EMOJI_RE.sub(" ", s)
    s = _SEPARATOR_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# --- json-safe conversion -------------------------------------------------

def _jsonify(value: Any) -> Any:
    """Convert numpy / NaN values into plain json-safe Python values."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if math.isnan(f):
            return None
        return int(f) if f.is_integer() else f
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    """Render a DataFrame row as a json-safe plain dict."""
    return {col: _jsonify(row[col]) for col in row.index}


# --- loading --------------------------------------------------------------

def _load_enrichment() -> pd.DataFrame:
    """Load the enrichment cache into a DataFrame keyed by Listing_ID."""
    path = config.DATA_DIR / "inventory_enriched.json"
    if not path.exists():
        return pd.DataFrame(columns=["Listing_ID", *_ENRICH_FIELDS])
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    records = []
    for lid, rec in raw.items():
        row = {"Listing_ID": int(lid)}
        for field in _ENRICH_FIELDS:
            row[field] = rec.get(field)
        records.append(row)
    return pd.DataFrame(records, columns=["Listing_ID", *_ENRICH_FIELDS])


def _build_inventory() -> pd.DataFrame:
    """Read the xlsx, merge enrichment, clean, and validate the inventory."""
    df = pd.read_excel(config.INVENTORY_PATH, sheet_name=SHEET_NAME)
    df["Listing_ID"] = df["Listing_ID"].astype(int)

    # Normalize text identity columns.
    for col in ("make", "model", "trim"):
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Merge the enrichment cache on Listing_ID.
    enrich = _load_enrichment()
    df = df.merge(enrich, on="Listing_ID", how="left")
    for field in _ENRICH_FIELDS:
        if field not in df.columns:
            df[field] = None

    # Cleaned, contact-free description (original column kept intact).
    df["description_clean"] = df["description"].apply(_clean_description)

    # Validate year (2000-2027): out-of-range -> NaN.
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    bad_year = df["year"].notna() & ~df["year"].between(YEAR_MIN, YEAR_MAX)
    df.loc[bad_year, "year"] = np.nan

    # Validate price (null or 10k-5M): out-of-range -> None.
    df["price_aed"] = pd.to_numeric(df["price_aed"], errors="coerce")
    bad_price = df["price_aed"].notna() & ~df["price_aed"].between(PRICE_MIN, PRICE_MAX)
    df.loc[bad_price, "price_aed"] = np.nan

    for field in ("monthly_payment_aed", "mileage_km"):
        df[field] = pd.to_numeric(df[field], errors="coerce")

    df["has_warranty"] = df["has_warranty"].fillna(False).astype(bool)
    return df


# Module-level cache — built once on first access.
_INVENTORY: pd.DataFrame | None = None


def load_inventory() -> pd.DataFrame:
    """Return the merged, cleaned inventory DataFrame (cached).

    Builds the inventory on first call and caches it at module level; later
    calls return the same DataFrame.
    """
    global _INVENTORY
    if _INVENTORY is None:
        _INVENTORY = _build_inventory()
    return _INVENTORY


# --- search / lookup ------------------------------------------------------

def search_cars(
    make: str | None = None,
    model: str | None = None,
    body_type: str | None = None,
    color: str | None = None,
    min_price: int | None = None,
    max_price: int | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    max_mileage: int | None = None,
    keywords: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search the inventory. All filters optional, case-insensitive.

    Args:
        make: Exact (case-insensitive) make match, e.g. "mercedes-benz".
        model: Substring (case-insensitive) match on model.
        body_type: Exact body type, e.g. "suv".
        color: Substring (case-insensitive) match on exterior color.
        min_price, max_price: Price bounds in AED. When either is given,
            cars with an unknown (null) price are EXCLUDED.
        min_year, max_year: Model-year bounds.
        max_mileage: Maximum mileage in km (excludes cars over the limit;
            cars with unknown mileage are kept).
        keywords: Whitespace-separated tokens; every token must appear
            (case-insensitive substring) in the title or cleaned description.
        limit: Maximum number of results.

    Returns:
        A list of json-safe car dicts.
    """
    df = load_inventory()
    mask = pd.Series(True, index=df.index)

    if make:
        mask &= df["make"] == make.strip().lower()
    if model:
        mask &= df["model"].str.contains(re.escape(model.strip().lower()), na=False)
    if body_type:
        mask &= df["body_type"].fillna("").str.lower() == body_type.strip().lower()
    if color:
        mask &= df["color"].fillna("").str.lower().str.contains(
            re.escape(color.strip().lower()), na=False
        )

    price_filtered = min_price is not None or max_price is not None
    if price_filtered:
        # Exclude unknown-price cars when a price filter is applied.
        mask &= df["price_aed"].notna()
        if min_price is not None:
            mask &= df["price_aed"] >= min_price
        if max_price is not None:
            mask &= df["price_aed"] <= max_price

    if min_year is not None:
        mask &= df["year"].notna() & (df["year"] >= min_year)
    if max_year is not None:
        mask &= df["year"].notna() & (df["year"] <= max_year)

    if max_mileage is not None:
        # Keep cars with unknown mileage; exclude those over the cap.
        mask &= df["mileage_km"].isna() | (df["mileage_km"] <= max_mileage)

    if keywords:
        haystack = (
            df["title"].fillna("").str.lower() + " " + df["description_clean"].str.lower()
        )
        for token in keywords.lower().split():
            mask &= haystack.str.contains(re.escape(token), na=False)

    results = df[mask].head(limit)
    return [_row_to_dict(row) for _, row in results.iterrows()]


def get_car_by_id(listing_id: int) -> dict[str, Any] | None:
    """Fetch a single car record by its ``Listing_ID``.

    Args:
        listing_id: The inventory identifier of the car.

    Returns:
        The json-safe car record, or ``None`` if not found.
    """
    df = load_inventory()
    match = df[df["Listing_ID"] == int(listing_id)]
    if match.empty:
        return None
    return _row_to_dict(match.iloc[0])

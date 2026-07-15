"""Lead capture and viewing bookings (pure code, no LLM).

Appends qualified sales leads to ``data/leads.csv`` and confirmed viewing
appointments to ``data/bookings.csv``. All slot validation happens here, in
code — the model only supplies structured arguments; it never decides whether
a booking is valid.
"""

from __future__ import annotations

import csv
from datetime import date as date_cls
from datetime import datetime
from pathlib import Path
from typing import Any

from app import config

#: CSV of confirmed viewing bookings.
BOOKINGS_PATH: Path = config.DATA_DIR / "bookings.csv"
#: CSV of captured leads (also referenced via config.LEADS_PATH).
LEADS_PATH: Path = config.LEADS_PATH

# Booking window: Monday–Saturday, 08:00–20:00 (Sunday rejected).
_OPEN_HOUR = 8
_CLOSE_HOUR = 20
_SUNDAY = 6  # datetime.weekday(): Monday=0 … Sunday=6

_BOOKINGS_FIELDS = [
    "booking_id", "user_id", "listing_id", "car_summary",
    "date", "time", "name", "phone", "created_at",
]
_LEADS_FIELDS = [
    "lead_id", "user_id", "timestamp", "price_range",
    "requirements", "name", "phone", "interested_listing_id",
]

_SLOT_ERROR = "viewings run Mon-Sat 8am-8pm"


def _next_id(path: Path, prefix: str) -> str:
    """Return a sequential id like ``BK0007`` based on existing CSV rows."""
    count = 0
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            count = max(sum(1 for _ in f) - 1, 0)  # minus header
    return f"{prefix}{count + 1:04d}"


def _append_row(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    """Append one row to a CSV, writing the header if the file is new."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _validate_slot(date: str, time: str) -> str | None:
    """Validate a requested viewing slot in code.

    Returns ``None`` if the slot is valid, otherwise a human-readable reason.
    Rules: date parses, is not in the past, falls on Mon–Sat, and the time is
    within 08:00–20:00.
    """
    try:
        d = datetime.strptime(date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return f"'{date}' is not a valid date (use YYYY-MM-DD)."
    try:
        t = datetime.strptime(time, "%H:%M").time()
    except (ValueError, TypeError):
        return f"'{time}' is not a valid time (use HH:MM)."

    if d < date_cls.today():
        return "that date is in the past."
    if d.weekday() == _SUNDAY:
        return "that date is a Sunday."
    if not (_OPEN_HOUR <= t.hour < _CLOSE_HOUR or (t.hour == _CLOSE_HOUR and t.minute == 0)):
        return "that time is outside opening hours."
    return None


def book_viewing(
    user_id: str,
    listing_id: int,
    car_summary: str,
    date: str,
    time: str,
    customer_name: str,
    phone: str | None = None,
) -> dict[str, Any]:
    """Validate and record a viewing/test-drive booking.

    Args:
        user_id: Identifier of the shopper.
        listing_id: Listing being viewed.
        car_summary: Short human summary of the car (for the CSV row).
        date: Requested date, ``YYYY-MM-DD``.
        time: Requested time, ``HH:MM`` (24h).
        customer_name: Name the booking is under.
        phone: Optional contact number.

    Returns:
        ``{"ok": True, "booking_id": ..., ...}`` on success, or
        ``{"ok": False, "error": <reason>, "message": <policy>}`` if the slot
        is invalid.
    """
    reason = _validate_slot(date, time)
    if reason is not None:
        return {"ok": False, "error": reason, "message": _SLOT_ERROR}

    booking_id = _next_id(BOOKINGS_PATH, "BK")
    row = {
        "booking_id": booking_id,
        "user_id": user_id,
        "listing_id": listing_id,
        "car_summary": car_summary,
        "date": date,
        "time": time,
        "name": customer_name,
        "phone": phone if phone is not None else "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _append_row(BOOKINGS_PATH, _BOOKINGS_FIELDS, row)
    return {
        "ok": True,
        "booking_id": booking_id,
        "listing_id": listing_id,
        "car_summary": car_summary,
        "date": date,
        "time": time,
        "name": customer_name,
    }


def save_lead(
    user_id: str,
    price_range: str,
    requirements: str,
    name: str | None = None,
    phone: str | None = None,
    interested_listing_id: int | None = None,
) -> dict[str, Any]:
    """Append a qualified lead to ``data/leads.csv``.

    Args:
        user_id: Identifier of the shopper.
        price_range: Stated budget, free text (e.g. "under 120k AED").
        requirements: Concrete needs, free text (e.g. "white family SUV").
        name: Optional name.
        phone: Optional contact number.
        interested_listing_id: Optional listing the shopper liked.

    Returns:
        ``{"ok": True, "lead_id": ...}``.
    """
    lead_id = _next_id(LEADS_PATH, "LD")
    row = {
        "lead_id": lead_id,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "price_range": price_range,
        "requirements": requirements,
        "name": name if name is not None else "",
        "phone": phone if phone is not None else "",
        "interested_listing_id": "" if interested_listing_id is None else interested_listing_id,
    }
    _append_row(LEADS_PATH, _LEADS_FIELDS, row)
    return {"ok": True, "lead_id": lead_id}

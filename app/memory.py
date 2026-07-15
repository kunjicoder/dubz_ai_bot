"""Persistent long-term user memory (SQLite).

Stores one row per user in ``data/users.db``: their name plus an open-ended
JSON preferences blob (budget, body types, makes, colors, liked listings,
free-text notes). Short-term conversation history lives in the agent's
in-memory session dict — this module is only the durable, cross-session part.

Pure code: no LLM calls here. The model turns natural language into the
structured ``updates`` dict; this module just merges and persists it.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from app import config

#: SQLite database file for user profiles.
DB_PATH = config.DATA_DIR / "users.db"

#: Preference keys that hold lists — merged by union rather than replacement.
_LIST_KEYS = {"preferred_body_types", "makes", "colors", "liked_listing_ids"}


def _connect() -> sqlite3.Connection:
    """Open a connection to the users database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db() -> None:
    """Create the ``users`` table if it does not already exist."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id          TEXT PRIMARY KEY,
                name             TEXT,
                preferences_json TEXT NOT NULL DEFAULT '{}',
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL
            )
            """
        )


def get_profile(user_id: str) -> dict[str, Any] | None:
    """Load a user's profile.

    Args:
        user_id: Identifier for the user.

    Returns:
        A dict ``{user_id, name, preferences, created_at, updated_at}`` where
        ``preferences`` is the decoded JSON dict, or ``None`` if unknown.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    if row is None:
        return None
    return {
        "user_id": row["user_id"],
        "name": row["name"],
        "preferences": json.loads(row["preferences_json"] or "{}"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _merge_preferences(existing: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Merge ``updates`` into ``existing`` preferences.

    New keys are added, list-valued keys are unioned (order-preserving,
    de-duplicated), and scalar keys are replaced.
    """
    merged = dict(existing)
    for key, value in updates.items():
        if key in _LIST_KEYS or isinstance(value, list):
            current = merged.get(key) or []
            if not isinstance(current, list):
                current = [current]
            incoming = value if isinstance(value, list) else [value]
            union = list(current)
            for item in incoming:
                if item not in union:
                    union.append(item)
            merged[key] = union
        else:
            merged[key] = value
    return merged


def upsert_profile(user_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Create or update a user's profile by merging ``updates``.

    ``name`` (if present) updates the name column; every other key is merged
    into the preferences blob using union-for-lists / replace-for-scalars
    semantics. Existing values are never blindly overwritten.

    Args:
        user_id: Identifier for the user.
        updates: Durable facts to persist (e.g. ``{"name": "Karti",
            "budget_max": 120000, "colors": ["white"], "notes": "..."}``).

    Returns:
        The full profile after the merge.
    """
    updates = dict(updates or {})
    name_update = updates.pop("name", None)

    existing = get_profile(user_id)
    now = _now()

    if existing is None:
        prefs = _merge_preferences({}, updates)
        name = name_update
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (user_id, name, preferences_json, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (user_id, name, json.dumps(prefs, ensure_ascii=False), now, now),
            )
    else:
        prefs = _merge_preferences(existing["preferences"], updates)
        name = name_update if name_update is not None else existing["name"]
        with _connect() as conn:
            conn.execute(
                "UPDATE users SET name = ?, preferences_json = ?, updated_at = ?"
                " WHERE user_id = ?",
                (name, json.dumps(prefs, ensure_ascii=False), now, user_id),
            )

    return get_profile(user_id)  # type: ignore[return-value]


# Ensure the schema exists as soon as the module is imported.
init_db()

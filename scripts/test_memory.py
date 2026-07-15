"""Stage 3 demo: long-term memory, lead capture, and booking (non-interactive).

Runs two sessions for the same user across a simulated restart to show that
short-term history is per-session (in memory) while preferences persist
(SQLite). Then confirms a different user gets no recall, and dumps the
resulting users.db row, leads.csv, and bookings.csv.

Run with:  ``uv run python scripts/test_memory.py``
"""

from __future__ import annotations

import os
import sqlite3
import sys

# Make ``app`` importable when run as a plain script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import agent, leads, memory  # noqa: E402


def _reset_demo_state() -> None:
    """Start from a clean slate so the demo output is reproducible."""
    with sqlite3.connect(memory.DB_PATH) as conn:
        conn.execute("DELETE FROM users WHERE user_id IN ('karti', 'guest2')")
    for path in (leads.LEADS_PATH, leads.BOOKINGS_PATH):
        if path.exists():
            path.unlink()
    agent.SESSIONS.clear()


def _say(user_id: str, session_id: str, message: str) -> None:
    """Send one message to the agent and print the exchange."""
    print(f"\nUSER ({user_id}): {message}")
    reply = agent.chat(session_id, message, user_id=user_id).reply
    print(f"DUBZ: {reply}")
    print("-" * 72)


def _dump_file(path) -> None:
    """Print a CSV file's contents, or note that it is empty/missing."""
    print(f"\n### {path.name}")
    if not path.exists():
        print("(none written)")
        return
    print(path.read_text(encoding="utf-8").strip() or "(empty)")


def main() -> None:
    """Run session A, simulate a restart, run session B, then dump state."""
    _reset_demo_state()

    print("=" * 72)
    print("SESSION A — user_id=karti, session_id=a1")
    print("=" * 72)
    _say("karti", "a1", "hi im karti, looking for a white SUV under 120k for my family")
    _say("karti", "a1", "the velar looks great, i love it")
    _say("karti", "a1", "book me a viewing for it saturday at 4pm")
    _say("karti", "a1", "actually can you also do sunday 9am?")

    # Simulate a process restart: short-term session memory is gone,
    # long-term SQLite profile remains.
    print("\n>>> [simulating restart: clearing in-memory SESSIONS] <<<")
    agent.SESSIONS.clear()

    print("\n" + "=" * 72)
    print("SESSION B — user_id=karti, session_id=b1 (after restart)")
    print("=" * 72)
    _say("karti", "b1", "hi, im back")
    _say("karti", "b1", "any new options for me?")

    print("\n" + "=" * 72)
    print("DIFFERENT USER — user_id=guest2, session_id=g1 (should NOT recall karti)")
    print("=" * 72)
    print("guest2 profile in DB:", memory.get_profile("guest2"))
    _say("guest2", "g1", "hi, any recommendations?")

    # ---- Final persisted state ------------------------------------------
    print("\n" + "=" * 72)
    print("PERSISTED STATE")
    print("=" * 72)

    print("\n### users.db row for 'karti'")
    with sqlite3.connect(memory.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM users WHERE user_id = 'karti'").fetchone()
    if row:
        for key in row.keys():
            print(f"  {key}: {row[key]}")
    else:
        print("(no row)")

    _dump_file(leads.LEADS_PATH)
    _dump_file(leads.BOOKINGS_PATH)


if __name__ == "__main__":
    main()

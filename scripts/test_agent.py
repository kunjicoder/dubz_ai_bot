"""CLI REPL for the agent (manual test harness).

Reads user messages line-by-line from stdin and prints the agent's replies,
interleaved with the tool-call logs the agent prints to stdout. A single
session id is taken from argv so multi-turn context is exercised.

Usage:
    uv run python scripts/test_agent.py <session_id>
    # then type messages, or pipe a scripted conversation:
    printf 'hi\\nsomething else\\n' | uv run python scripts/test_agent.py demo
"""

from __future__ import annotations

import os
import sys

# Make ``app`` importable when run as a plain script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import agent  # noqa: E402


def main() -> None:
    """Run a stdin-driven REPL against a single agent session."""
    session_id = sys.argv[1] if len(sys.argv) > 1 else "cli"
    print(f"=== Dubz agent REPL (session={session_id}) ===")
    print("Type a message and press Enter. Ctrl-D / empty EOF to quit.\n")

    for line in sys.stdin:
        user_message = line.strip()
        if not user_message:
            continue
        print(f"\nUSER: {user_message}")
        reply = agent.chat(session_id, user_message).reply
        print(f"\nDUBZ: {reply}\n")
        print("-" * 70)


if __name__ == "__main__":
    main()

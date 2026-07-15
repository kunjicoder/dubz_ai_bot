"""Search smoke tests (manual script).

Prints the enrichment coverage summary, then exercises the data layer with a
handful of representative searches so the behaviour can be eyeballed. This is
a manual script, not a pytest suite.

Run with:  ``uv run python scripts/test_data.py``
"""

from __future__ import annotations

import os
import sys

# Make ``app`` importable when run as a plain script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import data  # noqa: E402

COVERAGE_FIELDS = ["price_aed", "monthly_payment_aed", "mileage_km", "color", "body_type"]


def print_coverage() -> None:
    """Print count and % of listings with each non-null enriched field."""
    df = data.load_inventory()
    total = len(df)
    print("=== Enrichment coverage ===")
    for field in COVERAGE_FIELDS:
        n = int(df[field].notna().sum())
        pct = (n / total * 100) if total else 0.0
        print(f"  {field:<22} {n:>3}/{total}  ({pct:5.1f}%)")
    print()


def show(label: str, **kwargs) -> None:
    """Run one search and print a compact summary of the results."""
    results = data.search_cars(**kwargs)
    print(f"--- {label}  ({len(results)} result(s)) ---")
    for c in results:
        print(
            f"  #{c['Listing_ID']:<3} {c['year']} {c['make']} {c['model']}"
            f" | price={c['price_aed']} mileage={c['mileage_km']}"
            f" body={c['body_type']} color={c['color']}"
        )
    if not results:
        print("  (no matches)")
    print()


def main() -> None:
    """Print coverage, then run the six representative searches."""
    print_coverage()
    show("(a) make=mercedes-benz", make="mercedes-benz")
    show("(b) max_price=100000", max_price=100_000)
    show("(c) body_type=suv, max_price=150000", body_type="suv", max_price=150_000)
    show("(d) keywords=warranty", keywords="warranty")
    show("(e) min_year=2023", min_year=2023)
    show("(f) make=tesla (expected no results)", make="tesla")


if __name__ == "__main__":
    main()

"""One-time inventory enrichment pass.

The source workbook (``data/inventory.xlsx``, sheet ``"cleaned dataset"``) has
no structured price / mileage / color / body_type columns — those facts are
buried in free-text ``title`` + ``description``. This script runs a single
LLM extraction pass over every listing, in batches, and caches the structured
result to ``data/inventory_enriched.json`` keyed by ``Listing_ID``.

Properties:
    * The xlsx is never modified.
    * Resumable — listings already present in the JSON cache are skipped.
    * Robust — each batch is retried once; per-field validation drops
      out-of-range values to null with a warning.

Run with:  ``uv run python scripts/enrich_inventory.py``
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from typing import Any

import litellm
import pandas as pd

# Make ``app`` importable when run as a plain script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("enrich")
litellm.suppress_debug_info = True

BATCH_SIZE = 10
SHEET_NAME = "cleaned dataset"

# Per-request timeout (seconds). Large reasoning models (e.g. GLM via NIM) can
# occasionally hang; bound each call so a stuck request fails and is retried.
REQUEST_TIMEOUT = int(os.getenv("ENRICH_REQUEST_TIMEOUT", "90"))

ALLOWED_BODY_TYPES = {
    "sedan", "suv", "coupe", "convertible", "hatchback",
    "pickup", "van", "sports",
}

PRICE_MIN, PRICE_MAX = 10_000, 5_000_000
MILEAGE_MIN, MILEAGE_MAX = 0, 500_000

# LiteLLM provider prefix -> the env var holding that provider's API key.
# Lets ENRICH_MODEL_NAME point at any provider (Gemini, NVIDIA NIM, Groq, …)
# without code changes — just set the matching key in .env.
_PROVIDER_ENV = {
    "gemini": "GEMINI_API_KEY",
    "nvidia_nim": "NVIDIA_NIM_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

SYSTEM_PROMPT = """\
You extract structured facts about used cars from noisy marketplace listings \
(title + description). You output ONLY valid JSON. Follow these rules exactly:

* price_aed is the FULL vehicle price only. NEVER a monthly payment, down \
payment, deposit, or fee. If only a monthly figure is stated, price_aed is null.
* monthly_payment_aed: the lowest stated monthly installment, if any.
* Never guess or estimate price or mileage. If not stated -> null. Do NOT \
compute price from monthly payments.
* color = exterior color only (not interior).
* body_type may be inferred from make/model world knowledge (e.g. GLC -> suv, \
Sunny -> sedan) even if not stated.
* spec: GCC, US, Japanese, Korean, European, etc. if stated.
* has_warranty is true only if a warranty is mentioned as available/included.
"""

USER_PROMPT_TEMPLATE = """\
Extract facts for the following {n} car listings.

Return a JSON object of the form {{"results": [...]}} where "results" is an \
array with EXACTLY one object per listing, in any order, each shaped as:
{{
  "listing_id": int,                # echo back the listing's Listing_ID
  "price_aed": int|null,
  "monthly_payment_aed": int|null,
  "mileage_km": int|null,
  "color": str|null,
  "body_type": "sedan"|"suv"|"coupe"|"convertible"|"hatchback"|"pickup"|"van"|"sports"|null,
  "has_warranty": bool,
  "spec": str|null
}}

Listings:
{listings}
"""


def _render_listing(row: pd.Series) -> str:
    """Render a single listing into a compact block for the prompt."""
    return (
        f"- Listing_ID: {int(row['Listing_ID'])}\n"
        f"  title: {str(row['title']).strip()}\n"
        f"  description: {str(row['description']).strip()}"
    )


def _extract_json(text: str) -> Any:
    """Parse model output into JSON, tolerating markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        # strip ```json ... ``` fences
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()
    return json.loads(text)


def _coerce_int(value: Any) -> int | None:
    """Best-effort convert a value to int, else None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        return int(digits) if digits else None
    return None


def _validate(obj: dict[str, Any], known_ids: set[int]) -> dict[str, Any] | None:
    """Validate and normalize one extracted record.

    Returns the cleaned record, or None if it cannot be attributed to a
    known listing. Out-of-range numeric fields and invalid body types are
    coerced to None with a logged warning.
    """
    listing_id = _coerce_int(obj.get("listing_id"))
    if listing_id is None or listing_id not in known_ids:
        log.warning("dropping record with unknown listing_id: %r", obj.get("listing_id"))
        return None

    price = _coerce_int(obj.get("price_aed"))
    if price is not None and not (PRICE_MIN <= price <= PRICE_MAX):
        log.warning("listing %s: price_aed %s out of range -> null", listing_id, price)
        price = None

    mileage = _coerce_int(obj.get("mileage_km"))
    if mileage is not None and not (MILEAGE_MIN <= mileage <= MILEAGE_MAX):
        log.warning("listing %s: mileage_km %s out of range -> null", listing_id, mileage)
        mileage = None

    body_type = obj.get("body_type")
    if isinstance(body_type, str):
        body_type = body_type.strip().lower()
    if body_type not in ALLOWED_BODY_TYPES:
        if body_type not in (None, ""):
            log.warning("listing %s: body_type %r not allowed -> null", listing_id, body_type)
        body_type = None

    color = obj.get("color")
    color = color.strip() if isinstance(color, str) and color.strip() else None

    spec = obj.get("spec")
    spec = spec.strip() if isinstance(spec, str) and spec.strip() else None

    return {
        "listing_id": listing_id,
        "price_aed": price,
        "monthly_payment_aed": _coerce_int(obj.get("monthly_payment_aed")),
        "mileage_km": mileage,
        "color": color,
        "body_type": body_type,
        "has_warranty": bool(obj.get("has_warranty", False)),
        "spec": spec,
    }


def _call_model(batch: pd.DataFrame) -> list[dict[str, Any]]:
    """Send one batch to the enrichment model and return raw record dicts."""
    listings_block = "\n".join(_render_listing(row) for _, row in batch.iterrows())
    user_prompt = USER_PROMPT_TEMPLATE.format(n=len(batch), listings=listings_block)

    resp = litellm.completion(
        model=config.ENRICH_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
        timeout=REQUEST_TIMEOUT,  # fail a hung request instead of blocking forever
    )
    content = resp.choices[0].message.content or ""
    parsed = _extract_json(content)
    if isinstance(parsed, dict):
        # unwrap {"results": [...]} (or the first list-valued key)
        for key in ("results", "listings", "cars", "data"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
        for value in parsed.values():
            if isinstance(value, list):
                return value
        raise ValueError("no list found in JSON object response")
    if isinstance(parsed, list):
        return parsed
    raise ValueError(f"unexpected JSON type: {type(parsed).__name__}")


def _process_batch(batch: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Extract + validate one batch, with a single retry on failure.

    Returns a mapping of str(listing_id) -> validated record. On unrecoverable
    failure the batch is skipped (empty mapping) so the run can continue.
    """
    known_ids = {int(v) for v in batch["Listing_ID"].tolist()}
    for attempt in (1, 2):
        try:
            raw = _call_model(batch)
            out: dict[str, dict[str, Any]] = {}
            for rec in raw:
                if not isinstance(rec, dict):
                    continue
                cleaned = _validate(rec, known_ids)
                if cleaned is not None:
                    out[str(cleaned["listing_id"])] = cleaned
            return out
        except Exception as e:  # noqa: BLE001 — network/JSON errors, keep going
            msg = str(e).splitlines()[0][:160]
            if attempt == 1:
                log.warning("batch failed (attempt 1), retrying once: %s", msg)
                time.sleep(5)
            else:
                log.error("batch failed twice, skipping ids %s: %s", sorted(known_ids), msg)
    return {}


def _load_cache(path: str) -> dict[str, dict[str, Any]]:
    """Load the existing enriched JSON cache, or an empty dict."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(path: str, data: dict[str, dict[str, Any]]) -> None:
    """Persist the enriched cache, pretty-printed and key-sorted numerically."""
    ordered = dict(sorted(data.items(), key=lambda kv: int(kv[0])))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _print_coverage(df: pd.DataFrame, cache: dict[str, dict[str, Any]]) -> None:
    """Print count and % of listings with each non-null enriched field."""
    total = len(df)
    fields = ["price_aed", "monthly_payment_aed", "mileage_km", "color", "body_type"]
    print("\n=== Enrichment coverage ===")
    print(f"listings enriched: {len(cache)}/{total}")
    for field in fields:
        n = sum(1 for r in cache.values() if r.get(field) is not None)
        pct = (n / total * 100) if total else 0.0
        print(f"  {field:<22} {n:>3}/{total}  ({pct:5.1f}%)")


def _ensure_provider_key(model: str) -> None:
    """Fail loudly unless the API key for ``model``'s provider is set.

    The provider is taken from the LiteLLM model prefix (e.g. ``gemini/…`` ->
    ``GEMINI_API_KEY``, ``nvidia_nim/…`` -> ``NVIDIA_NIM_API_KEY``). Keys are
    read from the environment (``.env`` is loaded by ``app.config``).
    """
    provider = model.split("/", 1)[0] if "/" in model else "gemini"
    env_var = _PROVIDER_ENV.get(provider, "GEMINI_API_KEY")
    key = os.getenv(env_var)
    if not key or not key.strip():
        sys.exit(
            f"FATAL: {env_var} is missing or empty (required for model '{model}'). "
            "Set it in .env (see .env.example)."
        )


def main() -> None:
    """Run the batched enrichment pass and cache results to disk."""
    _ensure_provider_key(config.ENRICH_MODEL_NAME)

    df = pd.read_excel(config.INVENTORY_PATH, sheet_name=SHEET_NAME)
    df["Listing_ID"] = df["Listing_ID"].astype(int)

    cache_path = str(config.DATA_DIR / "inventory_enriched.json")
    cache = _load_cache(cache_path)
    done_ids = set(cache.keys())

    todo = df[~df["Listing_ID"].astype(str).isin(done_ids)].reset_index(drop=True)
    log.info("total=%d already_done=%d to_process=%d model=%s",
             len(df), len(done_ids), len(todo), config.ENRICH_MODEL_NAME)

    n_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(n_batches):
        batch = todo.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        ids = [int(x) for x in batch["Listing_ID"].tolist()]
        log.info("batch %d/%d — ids %s", i + 1, n_batches, ids)

        results = _process_batch(batch)
        cache.update(results)
        _save_cache(cache_path, cache)  # checkpoint after every batch (resumable)

        if i < n_batches - 1:
            time.sleep(random.uniform(3, 5))  # free-tier rate limiting

    _print_coverage(df, cache)
    print(f"\nSaved -> {cache_path}")


if __name__ == "__main__":
    main()

# Dubz AI Bot 🚗

A conversational car-shopping assistant for dubizzle's UAE used-car marketplace.
It grounds every answer in a real inventory via function-calling + pandas retrieval, remembers returning shoppers, and books viewings — served as a FastAPI backend with a Streamlit chat client.

---

## Architecture

```
┌──────────────┐   HTTP/JSON    ┌───────────────────────────┐
│  Streamlit   │  ───(httpx)──▶ │        FastAPI (app.main)  │
│  ui/app.py   │ ◀──────────────│  /chat /inventory /cars    │
│ (no app/ import)              │  /users /health            │
└──────────────┘                └─────────────┬─────────────┘
                                               │
                                     ┌─────────▼──────────┐
                                     │   Agent loop       │
                                     │   (app.agent)      │
                                     │  LiteLLM → Gemini  │
                                     │  tool-call loop    │
                                     └───┬──────┬──────┬───┘
                          search/details │      │      │ profile / lead / booking
                                   ┌──────▼──┐ ┌─▼────┐ ┌▼─────────┐
                                   │ data.py │ │memory│ │ leads.py │
                                   │ pandas  │ │SQLite│ │  CSV     │
                                   │ +xlsx   │ │users │ │ leads/   │
                                   │ +json   │ │ .db  │ │ bookings │
                                   └─────────┘ └──────┘ └──────────┘
```

The UI talks to the backend **only** over HTTP — it imports nothing from `app/`, and
the backend never imports Streamlit. That boundary keeps the client swappable and the
service independently deployable.

---

## Quick start

**Prerequisites**
- [uv](https://docs.astral.sh/uv/) (Python package/venv manager) — Python 3.11+ is installed automatically by uv.
- A **Gemini API key** (free tier works): https://aistudio.google.com/apikey

**Steps**

```bash
# 1. Clone
git clone <repo-url> Dubz_AI_Bot
cd Dubz_AI_Bot

# 2. Configure your key
cp .env.example .env
#   then edit .env and set:  GEMINI_API_KEY=your_key_here

# 3. Install dependencies (creates .venv from uv.lock)
uv sync

# 4. (OPTIONAL) Re-run the inventory enrichment pass.
#    NOT required — data/inventory_enriched.json is already committed, so the
#    app runs out of the box without spending any API quota on extraction.
#    uv run python scripts/enrich_inventory.py

# 5. Terminal 1 — start the backend
uv run uvicorn app.main:app --reload      # http://127.0.0.1:8000

# 6. Terminal 2 — start the chat client
uv run streamlit run ui/app.py            # http://localhost:8501
```

Open the Streamlit URL, type a request like *"a white SUV under 120k for my family"*,
and expand **🔧 behind the scenes** on any reply to watch the agent's tool calls. To
demo long-term memory: set a **User ID** in the sidebar, chat, then click **Start new
session** — the assistant greets you by name and recalls your saved preferences.

**Optional CLI harnesses** (no UI needed):

```bash
uv run python scripts/test_data.py      # inventory coverage + search smoke tests
uv run python scripts/test_agent.py s1  # agent REPL (reads messages from stdin)
uv run python scripts/test_memory.py    # two-session long-term-memory demo
```

---

## Why these choices

**Streamlit over a notebook.** A reactive chat UI is the honest way to demo a
conversational agent, and putting it behind an HTTP API forces a clean separation
between "the product" (backend: agent, retrieval, memory) and "a client" (UI) —
the UI could be replaced by a mobile app or WhatsApp without touching the core.

**No agent framework — a hand-rolled tool loop via LiteLLM.** The agentic loop is
~40 lines: call the model with tools, execute any tool calls against local code,
feed results back, repeat (capped at 5 rounds). This gives full control over
grounding and error handling, stays completely explainable, and is provider-agnostic
— LiteLLM let me swap models freely (the enrichment pass even ran across Gemini and
NVIDIA-hosted GLM when free-tier quota ran dry).

**Function-calling + pandas retrieval over vector RAG.** The inventory is small,
structured tabular data where constraints must be enforced *exactly* — "under 120k"
and "SUV" are filters, not fuzzy similarities. Pandas filtering guarantees the model
can only ever talk about cars the tools actually returned, so it cannot hallucinate
listings, prices, or specs. The dataset's catch — price/mileage/color/body-type are
buried in free-text descriptions, not columns — is solved by a **one-time, cached LLM
extraction pass** (`scripts/enrich_inventory.py` → `data/inventory_enriched.json`),
keeping the request-time path pure pandas with zero per-query LLM retrieval cost.

**SQLite for long-term memory, in-memory dict for short-term.** Durable preferences
(name, budget, body types, liked cars) belong in a store that survives restarts, so
they live in `users.db` keyed by `user_id`. Transient conversation history is just the
current session and lives in an in-memory `SESSIONS` dict keyed by `session_id`. In
production the session store would move to **Redis** (shared across worker processes,
with TTL eviction); the interface is small enough that it's a drop-in swap.

---

## How it works

**Request flow.** A message goes Streamlit → httpx → `POST /chat` → `agent.chat()`,
which loads the session history (inserting a system prompt with today's date and, for
a known user, their stored profile), then runs the LiteLLM tool loop. The model may
call `search_cars` / `get_car_details` (pandas over the merged inventory),
`update_user_profile` (SQLite upsert with union-merge semantics),
`save_lead` / `book_viewing` (CSV append). The final reply plus the list of tool calls
made is returned to the UI, which renders the tool calls in a transparency panel.
The **enrichment pipeline** reads the xlsx (`"cleaned dataset"` sheet), batches
listings to a cheap Gemini model with strict "never invent price/mileage" rules,
validates each field in code, and caches the result to JSON — the app then merges that
onto the inventory at load time and adds a contact-scrubbed `description_clean` column.

**Validation lives in code, not the prompt.** The model only ever supplies structured
arguments; every rule is enforced deterministically. Booking slots are validated in
`leads.py` (date parses, not in the past, Monday–Saturday only, 08:00–20:00) and
rejected with a structured error the model relays — the LLM cannot talk its way into a
Sunday booking. Price filters exclude unknown-price cars in `data.py`; numeric fields
are range-checked; the JSON returned to the client is NaN/`numpy`-scrubbed. Errors are
mapped cleanly at the HTTP edge: LLM failures → **502**, unknown ids → **404**, bad
input → **422**, and no raw traceback ever reaches the client.

**Future work (out of scope here).** An embeddings re-rank layer for genuinely fuzzy
queries ("something sporty but practical") on top of the exact filters; a Redis session
store for multi-worker deployments; streaming token responses for lower perceived
latency; authentication and per-user rate limiting; containerization (Docker Compose
for backend + UI); an eval harness measuring retrieval accuracy against a labelled
query set; and a background memory-extraction sweep that re-reads transcripts to catch
durable facts the agent missed in the moment, as a recall safety net.

---

## Project layout

```
app/
  config.py    # env + model config + paths
  data.py      # inventory load/merge/clean + search_cars / get_car_by_id (pandas)
  agent.py     # tool defs, system prompt, tool-executing chat loop (LiteLLM)
  memory.py    # SQLite user profiles (get/upsert, union-merge)
  leads.py     # lead + booking capture, in-code slot validation (CSV)
  main.py      # FastAPI app: /chat /inventory/search /cars /users /health
ui/
  app.py       # Streamlit client (httpx only — imports nothing from app/)
scripts/
  enrich_inventory.py  # one-time cached LLM extraction pass
  test_data.py / test_agent.py / test_memory.py  # manual harnesses
data/
  inventory.xlsx            # source dataset (committed)
  inventory_enriched.json   # cached extraction (committed — no quota needed to run)
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | `{user_id, session_id, message}` → `{reply, session_id, tool_calls}` |
| GET  | `/inventory/search` | query params mirror `search_cars` → `{count, cars}` |
| GET  | `/cars/{listing_id}` | full record, or 404 |
| GET  | `/users/{user_id}/profile` | stored profile, or 404 |
| GET  | `/health` | `{status: "ok"}` |

## Screenshots

<img width="962" height="571" alt="image" src="https://github.com/user-attachments/assets/8bd03f7e-b925-43f3-abab-49f7d93df4bd" />
<img width="1071" height="568" alt="image" src="https://github.com/user-attachments/assets/fa511753-6adf-4854-9f08-460e647eefee" />
<img width="437" height="222" alt="image" src="https://github.com/user-attachments/assets/79e4e2b7-74bf-4437-aeb7-bcd247902545" />
<img width="977" height="430" alt="image" src="https://github.com/user-attachments/assets/fcf88721-74e6-4d4b-8638-77125994c1b9" />




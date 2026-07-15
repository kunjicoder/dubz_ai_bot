"""FastAPI backend for the Dubz AI Bot.

Exposes the agent, inventory, and user-profile layers over HTTP. The Streamlit
client talks to this service exclusively via HTTP — it never imports ``app``,
and this service never imports Streamlit. All request/response shapes are
Pydantic models; errors are mapped to clean status codes (no tracebacks reach
the client).
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import agent, data, memory


# --------------------------------------------------------------------------
# Pydantic models
# --------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str


class ChatRequest(BaseModel):
    user_id: str = "guest"
    session_id: str | None = None
    message: str = Field(..., min_length=1)


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class SearchResponse(BaseModel):
    count: int
    cars: list[dict[str, Any]]


class ProfileResponse(BaseModel):
    user_id: str
    name: str | None = None
    preferences: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


# --------------------------------------------------------------------------
# App + lifespan
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the inventory cache and ensure the user DB exists on startup."""
    memory.init_db()
    data.load_inventory()  # build + cache the DataFrame so the first request is fast
    yield


app = FastAPI(title="Dubz AI Bot", version="1.0.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request, exc):  # noqa: ANN001
    """Last-resort handler so raw tracebacks never reach the client."""
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Run one agent turn and return the reply plus the tool calls it made."""
    session_id = req.session_id or uuid.uuid4().hex
    result = agent.chat(session_id, req.message, user_id=req.user_id)
    if result.error is not None:
        # LLM / backend failure — surface a clean 502, not the raw exception.
        raise HTTPException(
            status_code=502,
            detail="The assistant is temporarily unavailable. Please try again.",
        )
    return ChatResponse(
        reply=result.reply,
        session_id=session_id,
        tool_calls=[ToolCall(name=tc["name"], args=tc.get("args", {})) for tc in result.tool_calls],
    )


@app.get("/inventory/search", response_model=SearchResponse)
def inventory_search(
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
    limit: int = Query(5, ge=1, le=50),
) -> SearchResponse:
    """Direct inventory search (query params mirror ``data.search_cars``)."""
    cars = data.search_cars(
        make=make,
        model=model,
        body_type=body_type,
        color=color,
        min_price=min_price,
        max_price=max_price,
        min_year=min_year,
        max_year=max_year,
        max_mileage=max_mileage,
        keywords=keywords,
        limit=limit,
    )
    return SearchResponse(count=len(cars), cars=cars)


@app.get("/cars/{listing_id}")
def get_car(listing_id: int) -> dict[str, Any]:
    """Return the full record for a listing, or 404 if it does not exist."""
    car = data.get_car_by_id(listing_id)
    if car is None:
        raise HTTPException(status_code=404, detail=f"No car with listing_id {listing_id}.")
    return car


@app.get("/users/{user_id}/profile", response_model=ProfileResponse)
def get_user_profile(user_id: str) -> ProfileResponse:
    """Return a user's stored profile, or 404 if they have none."""
    profile = memory.get_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No profile for user '{user_id}'.")
    return ProfileResponse(**profile)

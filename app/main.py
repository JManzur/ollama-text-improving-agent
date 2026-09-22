import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from app.agent import ImproveAgent, Mode, Timings
from app.config import settings

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger("agent")


async def _warmup(agent: ImproveAgent) -> None:
    try:
        timings = await agent.warmup()
        logger.info("model warm, load took %d ms", timings.load_ms)
    except Exception as exc:
        logger.warning("warmup failed, the first request will load the model: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent = ImproveAgent(settings)
    app.state.agent = agent
    logger.info("agent ready, model=%s host=%s", agent.model, settings.ollama_host)

    # Backgrounded so /health answers while the model loads.
    task = asyncio.create_task(_warmup(agent)) if settings.warmup_on_start else None
    yield
    if task:
        task.cancel()


app = FastAPI(
    title="Text improving agent",
    description="Rewrites text for grammar, clarity, concision and professional tone.",
    version="1.0.0",
    lifespan=lifespan,
)


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if settings.agent_api_key and x_api_key != settings.agent_api_key:
        raise HTTPException(401, "Missing or invalid X-API-Key header.")


class ImproveRequest(BaseModel):
    text: str = Field(min_length=1)
    mode: Mode = Mode.PROFESSIONAL
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class ImproveResponse(BaseModel):
    improved: str
    model: str
    mode: Mode
    elapsed_ms: int
    timings: Timings


@app.post("/improve", dependencies=[Depends(require_api_key)])
async def improve(payload: ImproveRequest, request: Request) -> ImproveResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(422, "The text field is empty.")
    if len(text) > settings.max_input_chars:
        raise HTTPException(
            413,
            f"The text is {len(text)} characters, the limit is {settings.max_input_chars}.",
        )

    agent: ImproveAgent = request.app.state.agent
    started = time.perf_counter()
    try:
        result = await agent.improve(text, payload.mode, payload.temperature)
    except httpx.TimeoutException as exc:
        raise HTTPException(
            504, f"The model did not answer within {settings.request_timeout} seconds."
        ) from exc
    except (httpx.HTTPError, ConnectionError) as exc:
        logger.warning("ollama unreachable: %s", exc)
        raise HTTPException(502, "Ollama is unreachable.") from exc

    if result.timings.load_ms:
        logger.info("the model was reloaded, load took %d ms", result.timings.load_ms)

    return ImproveResponse(
        improved=result.text,
        model=agent.model,
        mode=payload.mode,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        timings=result.timings,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request) -> dict[str, str]:
    agent: ImproveAgent = request.app.state.agent
    try:
        pulled = await agent.is_ready()
    except Exception as exc:
        logger.warning("readiness check failed: %s", exc)
        raise HTTPException(503, "Ollama is unreachable.") from exc

    if not pulled:
        raise HTTPException(503, f"Model {agent.model} is not pulled yet.")

    return {"status": "ready", "model": agent.model}

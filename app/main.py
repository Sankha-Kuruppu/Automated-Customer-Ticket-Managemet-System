"""TensorForge 2.0 Phase 2 - RideEat ticket classification service (FastAPI).

Every prediction/job endpoint runs the contract's checks in this exact order:
    auth (401) -> content type (415) -> size (413) -> JSON parse (400) -> validation (422)
Handlers read the raw body themselves so that order can be guaranteed.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.jobs import JobStore, JobWorker
from app.model import TicketClassifier
from app.validation import validate_batch, validate_ticket

log = logging.getLogger("tensorforge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
METRICS_PATH = ROOT / "models" / "metrics.json"

KB = 1024
MB = 1024 * KB
LIMIT_PREDICT = 1 * MB
LIMIT_BATCH = 5 * MB
LIMIT_JOBS = 25 * MB
BATCH_MAX = 100
JOB_MAX = 5000
RESULTS_LIMIT_MAX = 5000
RETRY_AFTER = "2"


# --------------------------------------------------------------------------- responses
class JSONResp(Response):
    """ASCII-escaped JSON so no input (even lone surrogates) can break encoding."""
    media_type = "application/json"

    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=True, separators=(",", ":")).encode("ascii")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details=None, headers=None):
        self.status, self.code, self.message = status, code, message
        self.details, self.headers = details, headers or {}


def error_response(status: int, code: str, message: str, details=None, headers=None) -> JSONResp:
    err = {"code": code, "message": message}
    if details:
        err["details"] = details
    return JSONResp({"error": err}, status_code=status, headers=headers)


# --------------------------------------------------------------------------- state
class State:
    model: TicketClassifier | None = None
    load_error: str | None = None
    loaded = threading.Event()
    store: JobStore | None = None
    worker: JobWorker | None = None


S = State()


def _load_model():
    try:
        S.model = TicketClassifier()
        log.info("model loaded: version=%s sha256=%s", S.model.model_version, S.model.sha256[:12])
    except Exception as exc:  # health keeps reporting 503
        S.load_error = repr(exc)
        log.exception("model failed to load")
    finally:
        S.loaded.set()


def _predict_tickets(tickets: list[dict]) -> list[dict]:
    S.loaded.wait(timeout=120)
    if S.model is None:
        raise RuntimeError("model not available")
    preds = S.model.predict(tickets)
    out = []
    for t, p in zip(tickets, preds):
        item = {"ticket_id": t["ticket_id"]} if t.get("ticket_id") is not None else {}
        item.update(p)
        out.append(item)
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.environ.get("API_KEY", "").strip():
        log.warning("API_KEY is not set: all prediction and job endpoints will return 401")
    threading.Thread(target=_load_model, daemon=True, name="model-loader").start()
    S.store = JobStore(os.environ.get("JOBS_DB_PATH", "/tmp/tensorforge/jobs.db"))
    S.store.recover_after_restart()
    S.worker = JobWorker(S.store, _predict_tickets)
    S.worker.start()
    yield


app = FastAPI(
    title="TensorForge 2.0 - RideEat Ticket Classification API",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)


# --------------------------------------------------------------------------- middleware / handlers
@app.middleware("http")
async def request_id_and_safety(request: Request, call_next):
    rid = request.headers.get("x-request-id")
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        response = error_response(500, "internal_error", "Internal server error.")
    if rid:
        response.headers["X-Request-ID"] = rid[:128]
    return response


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return error_response(exc.status, exc.code, exc.message, exc.details, exc.headers)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    codes = {404: ("not_found", "Resource not found."), 405: ("method_not_allowed", "Method not allowed.")}
    code, msg = codes.get(exc.status_code, ("http_error", str(exc.detail)))
    headers = dict(exc.headers) if getattr(exc, "headers", None) else None
    return error_response(exc.status_code, code, msg, headers=headers)


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    log.exception("unhandled error")
    return error_response(500, "internal_error", "Internal server error.")


# --------------------------------------------------------------------------- request checks
def check_auth(request: Request):
    expected = os.environ.get("API_KEY", "").strip()
    supplied = request.headers.get("x-api-key")
    if supplied is None:
        auth = request.headers.get("authorization", "")
        scheme, _, token = auth.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            supplied = token
    if supplied is None or not supplied.strip():
        raise ApiError(401, "unauthorized", "Missing API key. Send X-API-Key or Authorization Bearer.",
                       headers={"WWW-Authenticate": "Bearer"})
    if not expected or not hmac.compare_digest(supplied.strip().encode("utf-8", "replace"),
                                               expected.encode("utf-8")):
        raise ApiError(401, "unauthorized", "Invalid API key.", headers={"WWW-Authenticate": "Bearer"})


async def read_json_body(request: Request, limit: int):
    # 415: content type
    ctype = request.headers.get("content-type", "")
    if ctype.split(";")[0].strip().lower() != "application/json":
        raise ApiError(415, "unsupported_media_type", "Content-Type must be application/json.")
    # 413: size (declared length first, then actual bytes while streaming)
    too_big = ApiError(413, "payload_too_large", "Request body exceeds the maximum allowed size.")
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > limit:
                raise too_big
        except ValueError:
            pass
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise too_big
        chunks.append(chunk)
    body = b"".join(chunks)
    # 400: JSON parse
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        raise ApiError(400, "malformed_json", "Request body is not valid JSON.")


def validation_error(details):
    return ApiError(422, "validation_error", "Request failed validation.", details=details)


def get_store() -> JobStore:
    return S.store


# --------------------------------------------------------------------------- endpoints
@app.get("/health")
async def health():
    if S.model is not None:
        return JSONResp({"status": "ok", "model_version": S.model.model_version, "model_loaded": True})
    return JSONResp({"status": "loading", "model_version": None, "model_loaded": False}, status_code=503)


@app.post("/predict")
async def predict(request: Request):
    check_auth(request)
    body = await read_json_body(request, LIMIT_PREDICT)
    errs = validate_ticket(body, require_id=False)
    if errs:
        raise validation_error(errs)
    result = await run_in_threadpool(_predict_tickets, [body])
    return JSONResp(result[0])


@app.post("/predict/batch")
async def predict_batch(request: Request):
    check_auth(request)
    body = await read_json_body(request, LIMIT_BATCH)
    errs = validate_batch(body, BATCH_MAX)
    if errs:
        raise validation_error(errs)
    t0 = time.perf_counter()
    preds = await run_in_threadpool(_predict_tickets, body["tickets"])
    return JSONResp({
        "predictions": preds,
        "meta": {"count": len(preds), "model_version": preds[0]["model_version"],
                 "processing_time_ms": int((time.perf_counter() - t0) * 1000)},
    })


@app.post("/batch/jobs")
async def submit_job(request: Request):
    check_auth(request)
    body = await read_json_body(request, LIMIT_JOBS)
    errs = validate_batch(body, JOB_MAX)
    if errs:
        raise validation_error(errs)
    version = S.model.model_version if S.model else None
    idem = request.headers.get("idempotency-key")
    idem = idem[:128] if idem else None
    # Keep only the fields the model needs (plus ticket_id) to keep stored jobs small.
    tickets = [{"ticket_id": t["ticket_id"], "channel": t["channel"], "subject": t.get("subject", ""),
                "text": t["text"]} for t in body["tickets"]]
    store = get_store()
    res, problem = await run_in_threadpool(store.submit, tickets, version or "unknown", idem)
    if problem == "full":
        raise ApiError(429, "too_many_jobs", "Job queue is full. Retry later.",
                       headers={"Retry-After": "30"})
    job_id = res.get("created") or res.get("existing")
    if res.get("created"):
        S.worker.wake.set()
    status = store.get(job_id)
    if status is None:  # expired/deleted between calls; extremely unlikely
        raise ApiError(404, "job_not_found", "No job with this id.")
    return JSONResp(status, status_code=202,
                    headers={"Location": f"/batch/jobs/{job_id}", "Retry-After": RETRY_AFTER})


def _job_or_error(job_id: str) -> dict:
    status = get_store().get(job_id)
    if status is None:
        raise ApiError(404, "job_not_found", "No job with this id.")
    if status["status"] == "expired":
        raise ApiError(410, "job_expired", "Job results have expired.")
    return status


@app.get("/batch/jobs/{job_id}")
async def job_status(job_id: str, request: Request):
    check_auth(request)
    status = _job_or_error(job_id)
    headers = {"Retry-After": RETRY_AFTER} if status["status"] in ("queued", "running") else None
    return JSONResp(status, headers=headers)


@app.delete("/batch/jobs/{job_id}")
async def delete_job(job_id: str, request: Request):
    check_auth(request)
    if not get_store().delete(job_id):
        raise ApiError(404, "job_not_found", "No job with this id.")
    return Response(status_code=204)


def _int_param(request: Request, name: str, default, lo: int, hi: int | None):
    raw = request.query_params.get(name)
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise validation_error([{"field": name, "issue": "must be an integer"}])
    if v < lo or (hi is not None and v > hi):
        rng = f"between {lo} and {hi}" if hi is not None else f">= {lo}"
        raise validation_error([{"field": name, "issue": f"must be {rng}"}])
    return v


@app.get("/batch/jobs/{job_id}/results")
async def job_results(job_id: str, request: Request):
    check_auth(request)
    offset = _int_param(request, "offset", 0, 0, None)
    limit = _int_param(request, "limit", None, 1, RESULTS_LIMIT_MAX)
    status = _job_or_error(job_id)
    if status["status"] != "succeeded":
        raise ApiError(409, "job_not_ready",
                       f"Job status is '{status['status']}'. Results are available once it is 'succeeded'.")
    preds = get_store().results(job_id) or []
    total = len(preds)
    if limit is None:
        limit = max(total - offset, 1)
    page = preds[offset:offset + limit]
    next_offset = offset + limit if offset + limit < total else None
    return JSONResp({
        "job_id": job_id,
        "status": "succeeded",
        "total": total,
        "offset": offset,
        "limit": limit,
        "next_offset": next_offset,
        "model_version": status["model_version"],
        "predictions": page,
    })


# --------------------------------------------------------------------------- demo UI (public, read-only)
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC / "index.html", media_type="text/html")


@app.get("/model/metrics", include_in_schema=False)
async def model_metrics():
    """Training/validation report shown on the dashboard. Contains no predictions or secrets."""
    try:
        with open(METRICS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        raise ApiError(404, "not_found", "Metrics not available.")
    return JSONResp(data)

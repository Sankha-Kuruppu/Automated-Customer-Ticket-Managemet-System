"""Asynchronous batch jobs: SQLite-backed store plus one background worker thread.

* State lives in SQLite (JOBS_DB_PATH), so it survives a process restart and is shared by
  anything that opens the same file.
* On startup, jobs that were `running` become `failed` with error.code = "interrupted";
  `queued` jobs are picked up again.
* Capacity: 1 running + up to 3 queued. More returns 429.
* Finished jobs are kept for RETENTION_HOURS (default 24, contract minimum 6), then return 410.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

MAX_ACTIVE_JOBS = 4  # 1 running + 3 queued
CHUNK_SIZE = 100
RETENTION_HOURS = float(os.environ.get("JOB_RETENTION_HOURS", "24"))

ACTIVE = ("queued", "running")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None


class JobStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                total INTEGER NOT NULL,
                processed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                expires_at TEXT,
                model_version TEXT,
                error_code TEXT,
                error_message TEXT,
                idempotency_key TEXT UNIQUE,
                tickets TEXT,
                results TEXT,
                seq INTEGER
            )"""
        )

    # -- helpers -----------------------------------------------------------------------
    def _q(self, sql: str, args=()):
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def _row(self, job_id: str):
        rows = self._q(
            "SELECT job_id,status,total,processed,created_at,started_at,finished_at,expires_at,"
            "model_version,error_code,error_message FROM jobs WHERE job_id=?",
            (job_id,),
        )
        return rows[0] if rows else None

    @staticmethod
    def _to_status(r) -> dict:
        return {
            "job_id": r[0],
            "status": r[1],
            "total": r[2],
            "processed": r[3],
            "created_at": r[4],
            "started_at": r[5],
            "finished_at": r[6],
            "expires_at": r[7],
            "model_version": r[8],
            "error": {"code": r[9], "message": r[10]} if r[1] == "failed" else None,
        }

    # -- lifecycle -----------------------------------------------------------------------
    def recover_after_restart(self):
        now = _iso(_now())
        expires = _iso(_now() + timedelta(hours=RETENTION_HOURS))
        self._q(
            "UPDATE jobs SET status='failed', finished_at=?, expires_at=?, error_code='interrupted',"
            " error_message='Service restarted while the job was running.', tickets=NULL"
            " WHERE status='running'",
            (now, expires),
        )

    def expire_old(self):
        now = _iso(_now())
        self._q(
            "UPDATE jobs SET status='expired', tickets=NULL, results=NULL"
            " WHERE expires_at IS NOT NULL AND expires_at <= ? AND status!='expired'",
            (now,),
        )

    # -- API operations ----------------------------------------------------------------
    def find_by_idempotency_key(self, key: str) -> dict | None:
        rows = self._q("SELECT job_id FROM jobs WHERE idempotency_key=?", (key,))
        return self.get(rows[0][0]) if rows else None

    def submit(self, tickets: list[dict], model_version: str, idempotency_key: str | None):
        """Returns (status_dict, None) or (None, "full")."""
        with self._lock:
            if idempotency_key:
                r = self._conn.execute("SELECT job_id FROM jobs WHERE idempotency_key=?",
                                       (idempotency_key,)).fetchone()
                if r:
                    return {"existing": r[0]}, None
            active = self._conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if active >= MAX_ACTIVE_JOBS:
                return None, "full"
            job_id = str(uuid.uuid4())
            seq = self._conn.execute("SELECT COALESCE(MAX(seq),0)+1 FROM jobs").fetchone()[0]
            self._conn.execute(
                "INSERT INTO jobs (job_id,status,total,processed,created_at,model_version,"
                "idempotency_key,tickets,seq) VALUES (?,?,?,?,?,?,?,?,?)",
                (job_id, "queued", len(tickets), 0, _iso(_now()), model_version, idempotency_key,
                 json.dumps(tickets, ensure_ascii=True), seq),
            )
        return {"created": job_id}, None

    def get(self, job_id: str) -> dict | None:
        self.expire_old()
        r = self._row(job_id)
        return self._to_status(r) if r else None

    def raw_status(self, job_id: str) -> str | None:
        r = self._q("SELECT status FROM jobs WHERE job_id=?", (job_id,))
        return r[0][0] if r else None

    def results(self, job_id: str) -> list[dict] | None:
        r = self._q("SELECT results FROM jobs WHERE job_id=?", (job_id,))
        return json.loads(r[0][0]) if r and r[0][0] else None

    def delete(self, job_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
            return cur.rowcount > 0

    # -- worker side -------------------------------------------------------------------
    def next_queued(self):
        with self._lock:
            r = self._conn.execute(
                "SELECT job_id, tickets FROM jobs WHERE status='queued' ORDER BY seq LIMIT 1").fetchone()
            if not r:
                return None
            self._conn.execute("UPDATE jobs SET status='running', started_at=? WHERE job_id=?",
                               (_iso(_now()), r[0]))
            return r[0], json.loads(r[1])

    def progress(self, job_id: str, processed: int) -> bool:
        """Update progress. Returns False if the job was deleted (cancelled) meanwhile."""
        with self._lock:
            cur = self._conn.execute("UPDATE jobs SET processed=? WHERE job_id=? AND status='running'",
                                     (processed, job_id))
            return cur.rowcount > 0

    def finish(self, job_id: str, results: list[dict] | None, error: tuple[str, str] | None = None):
        now = _now()
        with self._lock:
            if error:
                self._conn.execute(
                    "UPDATE jobs SET status='failed', finished_at=?, expires_at=?, error_code=?,"
                    " error_message=?, tickets=NULL WHERE job_id=? AND status='running'",
                    (_iso(now), _iso(now + timedelta(hours=RETENTION_HOURS)), error[0], error[1], job_id),
                )
            else:
                self._conn.execute(
                    "UPDATE jobs SET status='succeeded', processed=total, finished_at=?, expires_at=?,"
                    " results=?, tickets=NULL WHERE job_id=? AND status='running'",
                    (_iso(now), _iso(now + timedelta(hours=RETENTION_HOURS)),
                     json.dumps(results, ensure_ascii=True), job_id),
                )


class JobWorker(threading.Thread):
    def __init__(self, store: JobStore, predict_fn):
        super().__init__(daemon=True, name="job-worker")
        self.store = store
        self.predict_fn = predict_fn
        self.wake = threading.Event()

    def run(self):
        while True:
            try:
                job = self.store.next_queued()
            except Exception:
                job = None
            if job is None:
                self.wake.wait(timeout=1.0)
                self.wake.clear()
                continue
            job_id, tickets = job
            try:
                results: list[dict] = []
                cancelled = False
                for start in range(0, len(tickets), CHUNK_SIZE):
                    chunk = tickets[start:start + CHUNK_SIZE]
                    results.extend(self.predict_fn(chunk))
                    if not self.store.progress(job_id, len(results)):
                        cancelled = True
                        break
                    time.sleep(0)  # yield so /health and /predict stay responsive
                if not cancelled:
                    self.store.finish(job_id, results)
            except Exception as exc:  # never let the worker die
                self.store.finish(job_id, None, ("internal_error", f"Job failed: {type(exc).__name__}"))

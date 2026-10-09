import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_KEY = "test-key-for-local-runs-only"
os.environ["API_KEY"] = TEST_KEY
os.environ["JOBS_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="tf-jobs-"), "jobs.db")

SCHEMAS = ROOT / "schemas"


def _validator(name: str) -> Draft202012Validator:
    with open(SCHEMAS / f"{name}.schema.json", encoding="utf-8") as f:
        return Draft202012Validator(json.load(f), format_checker=FormatChecker())


VALIDATORS = {n: _validator(n) for n in (
    "predict_response", "batch_response", "batch_job_status", "batch_job_results",
    "error_response", "health_response",
)}


def assert_schema(name: str, body):
    errors = sorted(VALIDATORS[name].iter_errors(body), key=lambda e: e.path)
    assert not errors, f"{name} schema errors: " + "; ".join(f"{list(e.path)}: {e.message}" for e in errors)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import S, app

    with TestClient(app) as c:
        S.loaded.wait(timeout=180)
        assert S.model is not None, f"model failed to load: {S.load_error}"
        yield c


@pytest.fixture
def auth():
    return {"X-API-Key": TEST_KEY}

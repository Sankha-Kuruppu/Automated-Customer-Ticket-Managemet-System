---
title: RideEat Ticket Router
emoji: 🎫
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# RideEat Ticket Router — TensorForge 2.0 Phase 2

A multilingual customer-support ticket **classification and routing** service for the fictional
RideEat ride-hailing and food-delivery platform. For each ticket it predicts:

| Output | Meaning |
|---|---|
| `category` | primary issue (11 classes) |
| `secondary_category` | a second issue if clearly present, else `null` |
| `is_urgent` | needs immediate human attention |
| `team` | derived from `category` with the fixed routing table |
| `confidence` | calibrated probability of the primary category |
| `needs_human_review` | *bonus*: `true` when confidence is below the review threshold |

Tickets arrive in English, Sinhala, Tamil, Singlish, Tanglish and mixed script, over email, chat
and call transcripts. **No LLM is used**: the model is our own trained classifier (see *Model*).

---

## Quick start

```bash
docker build -t rideeat-router .
docker run -p 8000:8000 -e API_KEY=<key> rideeat-router
```

* Demo UI and model dashboard: <http://localhost:8000/>
* Interactive API docs: <http://localhost:8000/docs>
* The container needs **no internet at runtime**: the encoder weights are downloaded at *build*
  time (pinned commit) and baked into the image. `/health` returns 200 ~5–15 s after start.

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H "X-API-Key: <key>" -H "Content-Type: application/json" \
  -d '{"ticket_id":"T1","channel":"chat","text":"bro mage order eka hour ekakata late, refund ekak denna"}'
```

### Run without Docker (development)

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
python training/download_encoder.py           # one-time: fetch multilingual-e5-small
API_KEY=dev-key uvicorn app.main:app --port 8000
```

---

## API (contract v2.1.0)

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | public | `200 {"status":"ok"}` once the model is loaded, `503` while loading |
| `POST /predict` | key | one ticket |
| `POST /predict/batch` | key | 1–100 tickets, synchronous, atomic validation |
| `POST /batch/jobs` | key | 1–5,000 tickets, async → `202` + `Location` + `Retry-After` |
| `GET /batch/jobs/{id}` | key | job status / progress |
| `GET /batch/jobs/{id}/results` | key | results in request order, paged with `offset`/`limit` |
| `DELETE /batch/jobs/{id}` | key | cancel or discard → `204` |

**Authorization.** `X-API-Key: <key>` or `Authorization: Bearer <key>`. The expected key is read
from the `API_KEY` environment variable (never stored in the image or repo) and compared with
`hmac.compare_digest`. If `API_KEY` is unset, every protected endpoint returns `401` (the service
never runs open). The key is never logged.

**Check order** on every protected POST: `401 → 415 → 413 → 400 → 422`. The handlers read the
raw body themselves so an unauthorised request never reaches parsing, validation or the model.
Size limits: 1 MB (`/predict`), 5 MB (`/predict/batch`), 25 MB (`/batch/jobs`).
Unknown routes / methods return JSON `404` / `405`; no input can produce a 5xx.

**Async jobs.** State lives in SQLite (`JOBS_DB_PATH`), processed by one background worker thread
in chunks of 100 so `/health` and `/predict` stay responsive (measured p95 `/health` = 12–19 ms
during a 4,800-ticket job). Capacity: 1 running + 3 queued, then `429` + `Retry-After`.
`Idempotency-Key` returns the existing job. Finished jobs are kept 24 h (contract minimum 6 h),
then return `410`; unknown ids return `404`; results before success return `409`. On restart,
jobs that were `running` become `failed` with `error.code = "interrupted"`.

**Consistency rules** enforced in code (`app/model.py::decide`): team from the fixed table,
`secondary_category ≠ category`, spam is never urgent and never has a secondary label.

---

## Model

```
ticket ─┬─ normalise (NFC, mask URLs/IDs/numbers) ─ TF-IDF word 1-2g + char_wb 2-5g ─ LinearSVC (sigmoid-calibrated) ─┐
        │                                                                                                          ├─ weighted soft vote ─ thresholds + rules ─ response
        └─ "query: " + subject + text ─ multilingual-e5-small (mean-pooled, L2) ─ logistic regression ─────────────┘
```

Three heads (category, secondary-with-"none", urgency) share this structure. Ensemble weights,
decision thresholds and a temperature for confidence calibration are chosen on the validation set,
then the final model is refit on train + validation.

### How it was trained, validated and improved

All numbers are on the **800-ticket validation set** with models fit on the **4,000-ticket train set**
(`python -m training.train`, full log in `training/train_log.txt`, metrics in `models/metrics.json`).

| Step | Model | Accuracy | Macro-F1 | ECE |
|---|---|---|---|---|
| E1 baseline | word TF-IDF + Complement NB | 0.501 | 0.503 | 0.278 |
| E2 | word TF-IDF + LogReg | 0.528 | 0.562 | 0.155 |
| E3 | char n-gram TF-IDF + LogReg | 0.626 | 0.647 | 0.148 |
| E4 | word+char TF-IDF + LinearSVC (calibrated) | 0.646 | 0.676 | 0.061 |
| E5 | multilingual-e5 embeddings + LogReg | 0.690 | 0.702 | 0.030 |
| **E6** | **ensemble E4 + E5** (+ temperature 0.6) | **0.765** | **0.786** | see metrics |

Other heads on validation: urgency **F1 0.923** (P 0.947 / R 0.900); secondary-category
F1 0.622 (precision 0.949, recall 0.463 — tuned to avoid inventing second issues).

**Key finding.** 5-fold CV on train reaches **99 %** accuracy, but validation is only 61–65 %
for lexical models: train tickets are templated (one issue sentence wrapped in boilerplate), and
validation re-words the issue sentences. CV on train therefore over-estimates badly, so all model
selection uses the validation split. Removing cross-category boilerplate sentences did **not**
help (tested, 0.62–0.64); semantic sentence embeddings did, and they make *different* errors
than character n-grams (n-grams excel on Singlish at 0.96, embeddings on English/Sinhala/Tamil),
which is why the ensemble gains +11 points.

### Human-review threshold (bonus field)

`needs_human_review = confidence < 0.61`. On validation this flags ~31 % of tickets; the
auto-routed 69 % are **90 %** accurate versus 45 % for flagged ones. The threshold is the lowest
cut-off at which auto-routed accuracy reaches 90 %.

### Reproduce

```bash
python training/download_encoder.py
python -m training.train          # ~1–3 min on a laptop CPU, writes models/ticket_model.joblib
python -m pytest tests -q         # 71 contract tests, responses validated against schemas/
API_KEY=<key> BASE_URL=http://localhost:8000 python scripts/smoke_test.py   # end-to-end
```

`model_version` (e.g. `v1.1.0`) is stored inside `models/ticket_model.joblib`; its SHA-256 is
recorded in `models/metrics.json` and shown on the dashboard.

---

## Repository layout

```
app/            FastAPI service (main.py), validation, job store/worker, model + encoder, demo UI
training/       train.py (experiments + final model), finetune.py, download_encoder.py
models/         ticket_model.joblib, metrics.json (encoder weights are fetched at build time)
schemas/        official OpenAPI contract + JSON Schemas used by the tests
tests/          pytest contract tests
scripts/        smoke_test.py: end-to-end check against local or hosted URL
data/           RideEat synthetic train/validation sets
```

## Deployment (Hugging Face Spaces)

This README's front matter configures a Docker Space on port 8000. Create a public Docker Space,
add a **secret** `API_KEY`, and push this repository to it. The Space builds the same `Dockerfile`.

## Limitations

* Sinhala-script tickets are the weakest language (≈0.63 accuracy on validation).
* Secondary-category recall is modest (0.46); the threshold favours precision.
* On free hosting the SQLite job store lives on ephemeral disk: a full container *replacement*
  (not a process restart) loses job history.

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

## Model (v2.0.0)

```
                ┌─ normalise (NFC, mask URLs/IDs/numbers) ─ TF-IDF word 1-2g + char_wb 2-5g ─ LinearSVC (sigmoid-calibrated) ─┐
ticket ─────────┤                                                                                                           ├─ weighted soft vote ─ temperature ─ thresholds + rules ─ response
                └─ "query: " + subject + text ─ multilingual-e5-small, FINE-TUNED by us (multi-task) ─ linear heads ─────────┘
```

Three heads (category, secondary-with-"none", urgency) share this structure. The encoder is
`intfloat/multilingual-e5-small` (pinned commit) fine-tuned on our tickets with three heads at once;
its 96M-parameter vocabulary embedding matrix is frozen, so we only ship the changed transformer
layers as a 42 MB delta (`models/encoder_ft_delta.safetensors`) that the Docker build applies to the
pinned base model. Ensemble weights, decision thresholds and a calibration temperature are chosen on
validation, then every component is refit on train + validation.

### How it was trained, validated and improved

All numbers are on the **800-ticket validation set** with models fit on the **4,000-ticket train set**.
Logs: `training/train_log.txt`, `training/finetune_eval_log.txt`, `training/build_v2_log.txt`;
all metrics in `models/metrics.json` (also shown on the dashboard).

| Step | Model | Accuracy | Macro-F1 |
|---|---|---|---|
| E1 baseline | word TF-IDF + Complement NB | 0.501 | 0.503 |
| E2 | word TF-IDF + LogReg | 0.528 | 0.562 |
| E3 | char n-gram TF-IDF + LogReg | 0.626 | 0.647 |
| E4 | word+char TF-IDF + LinearSVC (calibrated) | 0.646 | 0.676 |
| E5 | frozen multilingual-e5 embeddings + LogReg | 0.690 | 0.702 |
| E6 (v1.1) | ensemble E4 + E5 | 0.765 | 0.786 |
| E7 | **fine-tuned** e5, multi-task, epoch 3 of 4, alone | 0.879 | 0.868 |
| **E8 (v2.0)** | **ensemble 0.1·E4 + 0.9·E7, temperature 0.5** | **0.874** | **0.861** |

Final v2.0 validation scores for every output:

| Output | Score |
|---|---|
| category | accuracy 0.874, macro-F1 0.861, **ECE 0.015** (well calibrated) |
| is_urgent | **F1 0.994** (precision 1.000, recall 0.988) |
| secondary_category | F1 0.840 (precision 0.900, recall 0.788); exact match incl. `null` 0.975 |
| consistency rules | 0 violations (team table, spam rule, secondary ≠ primary) |
| accuracy by language | en 0.875 · si 0.913 · ta 0.967 · singlish 0.780 · tanglish 0.950 · mixed 0.850 |

**Why E8 rather than E7?** Our selection rule takes, among ensemble weights within 0.01 macro-F1 of
the best, the one with the lowest log loss. Keeping 10 % TF-IDF costs 0.6 points of accuracy on
validation but gives the best-calibrated probabilities and keeps a second, independent view of the
text (character n-grams) as insurance, because the organisers say evaluation data may be
distributed differently.

**Key findings.**
1. 5-fold CV on train reaches **99 %**, but validation is only 61–65 % for lexical models. Train
   tickets are templated (one issue sentence wrapped in boilerplate), and validation rewords the
   issue sentence, so CV on train badly over-estimates. All selection therefore uses validation.
2. Removing cross-category boilerplate sentences did **not** help (0.62–0.64): the problem is
   unseen wording, not noise.
3. Meaning-based features generalise: frozen embeddings +4 points, an ensemble +12, and
   fine-tuning +23 points over the best lexical model.
4. Epochs (fine-tuned alone, validation macro-F1): 1 → 0.778, 2 → 0.845, **3 → 0.868**, 4 → 0.858.

### Human-review threshold (bonus field)

`needs_human_review = confidence < 0.35`, the lowest cut-off at which auto-routed tickets reach
90 % category accuracy on validation. It flags 3.75 % of tickets; flagged tickets are only 20 %
accurate, so they are the right ones to send to a person.

### Reproduce

```bash
python training/download_encoder.py                    # pinned base encoder
python -m training.train                                # E1-E6 experiments, v1 metrics
python -m training.finetune --mode eval --epochs 4      # ~10 min/epoch on a 12-core laptop CPU
python -m training.ensemble_eval                        # choose epoch, weights, thresholds
python -m training.finetune --mode final --epochs 3     # refit encoder on train + validation
python -m training.build_v2                             # final bundle -> models/ticket_model.joblib
python training/apply_delta.py                          # local run: builds models/encoder_ft
python -m pytest tests -q                               # 71 contract tests vs schemas/
API_KEY=<key> BASE_URL=http://localhost:8000 python scripts/smoke_test.py
```

`model_version` (`v2.0.0`) is stored inside `models/ticket_model.joblib`; the SHA-256 of the
bundle and of the encoder delta are recorded in `models/metrics.json` and shown on the dashboard.

---

## Repository layout

```
app/            FastAPI service (main.py), validation, job store/worker, model + encoder, demo UI
training/       train.py, finetune.py, ensemble_eval.py, build_v2.py, download_encoder.py, apply_delta.py, logs
models/         ticket_model.joblib, encoder_ft_delta.safetensors, metrics.json, eval histories
schemas/        official OpenAPI contract + JSON Schemas used by the tests
tests/          pytest contract tests
scripts/        smoke_test.py: end-to-end check against a local or hosted URL
docs/           report, video script, project guide
data/           RideEat synthetic train/validation sets
```

## Deployment (Hugging Face Spaces)

This README's front matter configures a Docker Space on port 8000. Create a public Docker Space,
add a **secret** `API_KEY`, and push this repository to it. The Space builds the same `Dockerfile`.

## Limitations

* Singlish is now the weakest language (0.78), whereas TF-IDF alone reached 0.96 there.
  A language-aware ensemble weight is the obvious next step.
* Data is synthetic; real tickets would need fresh validation.
* On free hosting the SQLite job store lives on ephemeral disk: a full container *replacement*
  (not a process restart) loses job history.
* The 5,000-ticket job takes ~3 min on a 12-core laptop; expect ~10–15 min on 2 vCPU
  (contract limit: 30 min).

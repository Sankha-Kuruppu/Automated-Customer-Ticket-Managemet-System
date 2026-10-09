"""Hand-written request validation that mirrors schemas/predict_request.schema.json exactly.

Returns lists of {"index"?, "field", "issue"} detail dicts, the format the contract requires.
"""
from __future__ import annotations

import re

from app.labels import CHANNELS

TEXT_MAX = 10_000
SUBJECT_MAX = 500
TICKET_ID_MAX = 64
_NON_WS = re.compile(r"\S")


def _is_str(v) -> bool:
    return isinstance(v, str)


def validate_ticket(t, require_id: bool, index: int | None = None) -> list[dict]:
    errs: list[dict] = []

    def err(field: str, issue: str):
        d = {"field": field, "issue": issue}
        if index is not None:
            d = {"index": index, **d}
        errs.append(d)

    if not isinstance(t, dict):
        err("ticket" if index is not None else "body", "must be a JSON object")
        return errs

    # ticket_id
    if "ticket_id" not in t or t["ticket_id"] is None:
        if require_id:
            err("ticket_id", "is required")
    elif not _is_str(t["ticket_id"]):
        err("ticket_id", "must be a string")
    elif len(t["ticket_id"]) > TICKET_ID_MAX:
        err("ticket_id", f"must be at most {TICKET_ID_MAX} characters")
    elif require_id and not t["ticket_id"]:
        err("ticket_id", "must not be empty")

    # channel
    if "channel" not in t or t["channel"] is None:
        err("channel", "is required")
    elif not _is_str(t["channel"]) or t["channel"] not in CHANNELS:
        err("channel", "must be one of email, chat, call_transcript")

    # subject (optional, but must be a string when present)
    if "subject" in t:
        if not _is_str(t["subject"]):
            err("subject", "must be a string")
        elif len(t["subject"]) > SUBJECT_MAX:
            err("subject", f"must be at most {SUBJECT_MAX} characters")

    # text
    if "text" not in t or t["text"] is None:
        err("text", "is required")
    elif not _is_str(t["text"]):
        err("text", "must be a string")
    elif not _NON_WS.search(t["text"]):
        err("text", "must contain at least one non-whitespace character")
    elif len(t["text"]) > TEXT_MAX:
        err("text", f"must be at most {TEXT_MAX} characters")

    return errs


def validate_batch(body, max_items: int) -> list[dict]:
    if not isinstance(body, dict):
        return [{"field": "body", "issue": "must be a JSON object with a 'tickets' array"}]
    tickets = body.get("tickets")
    if "tickets" not in body or tickets is None:
        return [{"field": "tickets", "issue": "is required"}]
    if not isinstance(tickets, list):
        return [{"field": "tickets", "issue": "must be an array"}]
    if not 1 <= len(tickets) <= max_items:
        return [{"field": "tickets", "issue": f"must contain between 1 and {max_items} items"}]

    errs: list[dict] = []
    seen: set[str] = set()
    for i, t in enumerate(tickets):
        item_errs = validate_ticket(t, require_id=True, index=i)
        errs.extend(item_errs)
        if isinstance(t, dict) and _is_str(t.get("ticket_id")):
            tid = t["ticket_id"]
            if tid in seen:
                errs.append({"index": i, "field": "ticket_id", "issue": "duplicate of an earlier item"})
            seen.add(tid)
    return errs

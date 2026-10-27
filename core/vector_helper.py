"""Shared helpers for content hashing and model-name keys.

Every place that needs to hash text or build a model key uses these functions
— write-time embedding, /health, search backfill, manual backfill, and the
pending-vectors table. There is exactly one implementation, so the hash and
model name can never drift between call sites.

SQLite has no SHA-256() function; we hash in Python and compare the hex
digests as TEXT.
"""
import hashlib


def content_hash(text: str) -> str:
    """SHA-256 hex digest of the UTF-8 encoding of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def model_key(model_name: str) -> str:
    """Return *model_name* unchanged (it already carries the dimension).

    Kept as a function so callers go through one gate; if the format ever
    changes, only this file needs to change.
    """
    return model_name


def is_dense_eligible(row: dict, exclude_lines: tuple) -> bool:
    """A record is dense-eligible when it is valid, in the right space, and
    its line is not in *exclude_lines*.

    The criteria are:
      - valid == 1  (core_records.valid)
      - line not in exclude_lines  (or line is NULL)
      - space_id matches  (checked by the caller; this function checks row-level)

    Returns True when the record should have a vector.
    """
    if not row.get("valid"):
        return False
    line = row.get("line")
    if line is not None and line in exclude_lines:
        return False
    return True
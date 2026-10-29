"""/health reports the image's content fingerprint as `build` (see scripts/tree_digest.py, scripts/verify_deploy.py).

It is read from the file the image build stored. Without the file (a local run) it is null; anything in the file that is not
exactly twelve lowercase hex characters is treated as no fingerprint and never echoed.
"""
import os
import uuid

_TMPROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".pytest_tmp"))
os.makedirs(_TMPROOT, exist_ok=True)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import main as app_main  # noqa: E402
from app.config import Settings  # noqa: E402


def _settings():
    return Settings(db_path=os.path.join(_TMPROOT, f"bd_{uuid.uuid4().hex}.db"), api_key="k", recall_paths=("keyword",),
                    rerank_enabled=False, rerank_n=200, store_busy_timeout_ms=5000, store_write_lock_timeout_ms=10000,
                    embed_max_concurrency=2, embed_batch=32, embed_backfill_wait_s=60.0, embed_acquire_s=30.0,
                    bge_small_dir=None, neighbour_candidates=False)


def _health(monkeypatch, tmp_path, content):
    path = tmp_path / "BUILD_DIGEST"
    if content is not None:
        path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    monkeypatch.setattr(app_main, "BUILD_DIGEST_PATH", str(path))
    with TestClient(app_main.make_app(_settings())) as client:
        r = client.get("/health")
    assert r.status_code == 200
    return r.json(), r.text


def test_the_fingerprint_the_image_stored_is_reported_as_build(monkeypatch, tmp_path):
    body, _ = _health(monkeypatch, tmp_path, "0123456789ab\n")
    assert body["build"] == "0123456789ab" and body["ok"] is True


def test_without_a_stored_fingerprint_build_is_null(monkeypatch, tmp_path):
    body, _ = _health(monkeypatch, tmp_path, None)
    assert "build" in body and body["build"] is None


@pytest.mark.parametrize("junk", ["", "\n", "0123456789a", "0123456789abc", "0123456789AB", "0123456789ag", "build=0123456789ab",
                                  "0123456789ab\n0123456789ab", b"\xff\xfe\x00\x01"])
def test_anything_that_is_not_a_twelve_hex_fingerprint_is_no_fingerprint(monkeypatch, tmp_path, junk):
    body, text = _health(monkeypatch, tmp_path, junk)
    assert body["build"] is None
    if isinstance(junk, str) and junk.strip():
        assert junk.strip() not in text, "garbage from the file must never be echoed"


def test_the_default_path_is_next_to_the_app_directory_where_the_image_stores_it():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.abspath(app_main.BUILD_DIGEST_PATH) == os.path.join(root, "BUILD_DIGEST")


def test_the_rest_of_health_is_unchanged(monkeypatch, tmp_path):
    body, _ = _health(monkeypatch, tmp_path, "0123456789ab")
    # pending_vectors left /health (it needed a full read of every space): scripts/count_missing_vectors.py has it
    assert set(body) == {"ok", "config", "build"}

"""Settings validation. Since D, multi-path fusion is allowed (any
combination of recall paths); unknown paths and bad numeric values still fail
fast instead of breaking halfway through a request.
"""
import pytest

from app.config import Settings


def _base(**ov):
    d = dict(db_path="x.db", api_key="k", recall_paths=("keyword",),
             rerank_enabled=False, rerank_n=200, store_busy_timeout_ms=5000, store_write_lock_timeout_ms=10000,
             embed_max_concurrency=2, embed_batch=32, embed_backfill_wait_s=60.0,
             embed_acquire_s=30.0, bge_small_dir=None, neighbour_candidates=False)
    d.update(ov)
    return Settings(**d)


def test_stem_english_default_off_and_toggle():
    assert _base().stem_english is False, "STEM_ENGLISH 默认关"
    assert _base(stem_english=True).stem_english is True


def test_unknown_recall_path_rejected():
    with pytest.raises(ValueError, match="unknown recall path"):
        _base(recall_paths=("bogus",)).validate()


def test_multi_path_allowed_since_d():
    # D lifted the one-path guard: any combination of the paths must validate.
    s = _base(recall_paths=("keyword", "dense"))
    assert s.validate() is s


def test_bad_numbers_rejected():
    with pytest.raises(ValueError):
        _base(embed_max_concurrency=0).validate()
    with pytest.raises(ValueError):
        _base(rerank_n=0).validate()


def test_negative_timeouts_rejected():
    with pytest.raises(ValueError, match="STORE_BUSY_TIMEOUT_MS"):
        _base(store_busy_timeout_ms=-1).validate()
    with pytest.raises(ValueError, match="STORE_WRITE_LOCK_TIMEOUT_MS"):
        _base(store_write_lock_timeout_ms=-10).validate()
    with pytest.raises(ValueError, match="EMBED_BACKFILL_WAIT_S"):
        _base(embed_backfill_wait_s=-1.0).validate()
    with pytest.raises(ValueError, match="EMBED_ACQUIRE_S"):
        _base(embed_acquire_s=-0.5).validate()


def test_negative_backfill_limits_rejected():
    with pytest.raises(ValueError, match="DENSE_BACKFILL_M"):
        _base(dense_backfill_m=-1).validate()
    with pytest.raises(ValueError, match="DENSE_BACKFILL_T"):
        _base(dense_backfill_t=-0.5).validate()

@pytest.mark.parametrize("field,value,name", [
    ("dense_query_t", 0, "DENSE_QUERY_T"),
    ("search_heavy_concurrency", -1, "SEARCH_HEAVY_CONCURRENCY"),
    ("search_heavy_wait_s", 0, "SEARCH_HEAVY_WAIT_S"),
    ("search_cache_mib", 0, "SEARCH_CACHE_MIB"),
])
def test_dense_index_settings_out_of_range_are_rejected(field, value, name):
    with pytest.raises(ValueError, match=name):
        _base(**{field: value}).validate()


def test_dense_index_settings_are_read_from_the_environment(monkeypatch):
    from app.config import load
    for k, v in {"SEARCH_HEAVY_CONCURRENCY": "3", "SEARCH_HEAVY_WAIT_S": "7.5", "SEARCH_CACHE_MIB": "300",
                 "DENSE_QUERY_T": "1.5"}.items():
        monkeypatch.setenv(k, v)
    s = load()
    assert (s.search_heavy_concurrency, s.search_heavy_wait_s, s.search_cache_mib, s.dense_query_t) == (3, 7.5, 300, 1.5)

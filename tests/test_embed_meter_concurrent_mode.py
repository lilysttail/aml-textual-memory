"""The embedding meter's concurrent mode — so the ruler can measure what a deployed service sees under many simultaneous searches.

The meter normally puts every provider request through the run's paid gate: one lock, one request outside at a time. That is right for
the answering / judging runs, and it makes any concurrency measurement of the embedding path meaningless: 16 searches wait their turn
behind one another (measured: ~1.1 s per request, 17 s per search), nothing like a service whose requests go out in parallel.

`concurrent=True` lets the requests overlap. The wall stays: before a request leaves, the upper bound of its tokens is ADDED to a running sum checked against
the cap (never taken off again, even if the reply is cheaper), so however the threads interleave, the sum of the upper bounds of everything
sent is at most the cap, and the real tokens are at most that. It cannot be combined with a spending meter (that one needs the serial gate).
"""
import json
import threading
import time

import pytest

from eval.ruler import dense as dense_mod
from eval.ruler import embed_meter

BODY = json.dumps({"model": "m", "input": ["abcdefgh"]}).encode()          # upper bound 8 + 2 = 10 tokens


def reply(tokens):
    return 200, json.dumps({"usage": {"total_tokens": tokens}, "data": []}).encode()


class Overlap:
    """A transport that records how many requests are inside it at once (and for how long it holds them)."""

    def __init__(self, hold_s=0.15, result=None):
        self.hold_s, self.result = hold_s, result or reply(4)
        self.now = self.peak = self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, url, headers, body, timeout):
        with self._lock:
            self.now += 1
            self.calls += 1
            self.peak = max(self.peak, self.now)
        time.sleep(self.hold_s)
        with self._lock:
            self.now -= 1
        return self.result


def _fire(meter, n):
    out = []

    def one():
        try:
            out.append(meter.transport("u", {}, BODY, 5))
        except Exception as e:                                         # noqa: BLE001
            out.append(e)
    ts = [threading.Thread(target=one) for _ in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return out


def test_by_default_requests_still_go_out_one_at_a_time():
    real = Overlap()
    _fire(embed_meter.EmbedMeter(real, cap_tokens=10 ** 6), 6)
    assert real.peak == 1


def test_in_concurrent_mode_requests_really_overlap():
    real = Overlap()
    out = _fire(embed_meter.EmbedMeter(real, cap_tokens=10 ** 6, concurrent=True), 8)
    assert real.peak >= 6, f"only {real.peak} of 8 requests were in flight together"
    assert all(o == reply(4) for o in out)


def test_the_summed_bounds_never_pass_the_cap_however_the_threads_interleave():
    real = Overlap(hold_s=0.05)
    m = embed_meter.EmbedMeter(real, cap_tokens=50, concurrent=True)         # five requests of bound 10 fit, no sixth
    out = _fire(m, 30)
    sent = [o for o in out if o == reply(4)]
    refused = [o for o in out if isinstance(o, embed_meter.CapReached)]
    assert len(sent) == 5 and len(refused) == 25 and real.calls == 5
    assert m.snapshot()["refused_after_cap"] == 25 and m.cap_reached.is_set()


def test_a_cheaper_reply_does_not_take_its_bound_off_the_sum():
    real = Overlap(hold_s=0, result=reply(1))
    m = embed_meter.EmbedMeter(real, cap_tokens=20, concurrent=True)
    m.transport("u", {}, BODY, 5)
    m.transport("u", {}, BODY, 5)
    with pytest.raises(embed_meter.CapReached):
        m.transport("u", {}, BODY, 5)
    assert m.snapshot()["tokens_by_replies"] == 2                          # what was really charged is still booked


def test_replies_are_booked_like_in_the_serial_mode():
    replies = iter([reply(7), (429, b"{}"), (200, b"not json")])
    m = embed_meter.EmbedMeter(lambda *a: next(replies), cap_tokens=10 ** 6, concurrent=True)
    for _ in range(3):
        m.transport("u", {}, BODY, 5)
    s = m.snapshot()
    assert s["calls"] == 3 and s["http_status"] == {"200": 2, "429": 1}
    assert s["tokens_by_replies"] == 7 + 10 and s["replies_without_usage"] == 2     # a 2xx without usage is booked at the bound; a 429 is not


def test_a_transport_exception_is_booked_at_the_bound_and_raised_again():
    def boom(*a):
        raise TimeoutError("t")
    m = embed_meter.EmbedMeter(boom, cap_tokens=10 ** 6, concurrent=True)
    with pytest.raises(TimeoutError):
        m.transport("u", {}, BODY, 5)
    s = m.snapshot()
    assert s["transport_exceptions"] == {"TimeoutError": 1} and s["tokens_charged_without_reply"] == 10


def test_a_reply_that_charges_more_than_its_bound_closes_everything():
    m = embed_meter.EmbedMeter(lambda *a: reply(99), cap_tokens=10 ** 6, concurrent=True)
    m.transport("u", {}, BODY, 5)
    assert m.snapshot()["replies_above_bound"] == 1
    with pytest.raises(embed_meter.CapReached):
        m.transport("u", {}, BODY, 5)


def test_concurrent_mode_cannot_be_combined_with_a_spending_meter():
    with pytest.raises(ValueError, match="concurrent"):
        embed_meter.EmbedMeter(Overlap(), cap_tokens=10, cost_meter=object(), price_key="x", concurrent=True)


def test_the_dense_config_carries_the_switch_and_defaults_to_the_serial_gate():
    assert dense_mod.DenseConfig().concurrent_embed is False
    assert dense_mod.DenseConfig(concurrent_embed=True).concurrent_embed is True


def test_the_load_command_has_the_switch_and_hands_it_to_the_dense_config():
    from eval.ruler import qa_main
    parser = qa_main.build_parser()
    base = ["load", "--data", "x", "--out", "y", "--search", "dense"]
    assert qa_main._dense_config(parser.parse_args(base)).concurrent_embed is False
    assert qa_main._dense_config(parser.parse_args(base + ["--concurrent-embed"])).concurrent_embed is True


def test_the_service_target_builds_its_meter_in_the_mode_the_config_asks_for(tmp_path, monkeypatch):
    from eval.ruler import target
    from eval.ruler.testing import FAKE_KEY, FakeEmbedApi
    monkeypatch.setenv("DASHSCOPE_API_KEY", FAKE_KEY)
    for flag in (False, True):
        t = target.ServiceTarget(str(tmp_path / f"{flag}.sqlite"), neighbours=False,
                                 dense=dense_mod.DenseConfig(transport=FakeEmbedApi(), concurrent_embed=flag))
        try:
            assert t.embed_meter._concurrent is flag
        finally:
            t.close()

"""step 4, part 2: counting that the meaning-based path really RAN (not just "was not skipped"): `DenseUseCounter` around
`core.recall.DenseRecall.candidates`, and the per-question `dense_used` in the contexts the ruler builds. Fake embedder, fake key."""
import threading

import pytest

from core import recall
from eval.ruler import corpus, dense, sampling, shapes, target
from eval.ruler.testing import FAKE_KEY, FakeEmbedApi, write_mini_corpus


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", FAKE_KEY)


@pytest.fixture()
def mini(tmp_path):
    return corpus.load_corpus(write_mini_corpus(str(tmp_path / "copy")), require_pin=False)


def fill(t, mini, sample="mini-1", shape="c"):
    for conv in mini.conversations:
        if conv.sample_id == sample:
            for b in corpus.chunk_conversation(conv):
                assert t.add(b.user_id, b.session_id, b.request_id, [shapes.render(shape, m, s) for m, s in b.pairs()]) == 200


class DownApi(FakeEmbedApi):
    """An embedding provider that answers 500 to everything: the dense half of every search is skipped, the keyword half still answers."""

    def __call__(self, url, headers, body, timeout):
        self.calls += 1
        return 500, b"{}"


# ---- the counter ----

def test_a_dense_target_counts_every_dense_search_and_how_many_returned_something(tmp_path, mini):
    original = recall.DenseRecall.candidates
    t = target.ServiceTarget(str(tmp_path / "d.sqlite"), neighbours=False, dense=dense.DenseConfig(transport=FakeEmbedApi()))
    fill(t, mini)
    assert t.dense_use is not None and (t.dense_use.calls, t.dense_use.nonempty) == (0, 0)
    user = corpus.user_id_of("mini-1")
    t.search(user, "wedding ceremony", 5)
    t.search(user, "another question about the dog", 5)
    assert (t.dense_use.calls, t.dense_use.nonempty) == (2, 2)
    t.close()
    assert recall.DenseRecall.candidates is original                      # the wrapper is gone with the target


def test_a_keyword_target_has_no_counter_and_nothing_is_wrapped(tmp_path, mini):
    original = recall.DenseRecall.candidates
    t = target.ServiceTarget(str(tmp_path / "k.sqlite"), neighbours=False)
    assert t.dense_use is None and recall.DenseRecall.candidates is original
    t.close()


def test_a_search_whose_query_vector_was_skipped_is_a_call_that_returned_nothing(tmp_path, mini):
    t = target.ServiceTarget(str(tmp_path / "d.sqlite"), neighbours=False, dense=dense.DenseConfig(transport=DownApi()))
    fill(t, mini)
    t.search(corpus.user_id_of("mini-1"), "wedding ceremony", 5)
    assert t.dense_use.calls == 1 and t.dense_use.nonempty == 0
    t.close()


def test_counting_does_not_change_what_a_search_returns(tmp_path, mini):
    user = corpus.user_id_of("mini-1")
    t = target.ServiceTarget(str(tmp_path / "d.sqlite"), neighbours=False, dense=dense.DenseConfig(transport=FakeEmbedApi()))
    fill(t, mini)
    with_counter = [h.id for h in t.search(user, "wedding ceremony", 5)]
    t.dense_use.uninstall()
    without = [h.id for h in t.search(user, "wedding ceremony", 5)]
    t.close()
    assert with_counter == without and with_counter


def test_two_counters_at_once_both_count_and_the_original_comes_back_when_the_last_one_leaves():
    original = recall.DenseRecall.candidates
    a, b = dense.DenseUseCounter().install(), dense.DenseUseCounter().install()
    assert recall.DenseRecall.candidates is not original
    a.uninstall()
    assert recall.DenseRecall.candidates is not original              # b still needs it
    b.uninstall()
    assert recall.DenseRecall.candidates is original
    b.uninstall()                                                      # a second uninstall is harmless
    assert recall.DenseRecall.candidates is original


def test_the_counts_are_right_under_many_threads():
    c = dense.DenseUseCounter()
    ts = [threading.Thread(target=lambda: [c._record(3) for _ in range(500)]) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert (c.calls, c.nonempty) == (4000, 4000)
    c._record(0)
    c._record(None)                                                   # a call that raised
    assert (c.calls, c.nonempty, c.raised) == (4002, 4000, 1)


# ---- the contexts the ruler builds ----

def _qids(mini):
    return [q["qa_id"] for q in mini.questions if q["sample_id"] == "mini-1"][:2]


def test_each_dense_context_says_whether_the_dense_path_ran_for_that_question(mini):
    ids = _qids(mini)
    stats = {}
    ctxs = sampling.build_contexts(mini, ids, shape="c", dense=dense.DenseConfig(transport=FakeEmbedApi()), stats_out=stats)
    assert all(ctxs[i].dense_used is True for i in ids)
    assert stats["dense_use"] == {"calls": len(ids), "nonempty": len(ids), "raised": 0}


def test_a_question_whose_dense_half_was_skipped_is_marked_not_used(mini):
    ids = _qids(mini)
    ctxs = sampling.build_contexts(mini, ids, shape="c", dense=dense.DenseConfig(transport=DownApi()))
    assert all(ctxs[i].dense_used is False for i in ids)


def test_keyword_contexts_carry_no_dense_flag(mini):
    ctxs = sampling.build_contexts(mini, _qids(mini), shape="c", dense=None)
    assert all(c.dense_used is None for c in ctxs.values())

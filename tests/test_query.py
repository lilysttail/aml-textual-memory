"""kw-query — tests for _fts_query term selection (written RED first).

Problem: the baseline took only the first 20 tokens of the query. On the
SWE-ContextBench related tasks the query is a PR description whose opening is
boilerplate ("This comments are hidden when you submit the pull request ..."),
so the real content words never made it into the FTS query. This is a generic
bug in keyword selection, not something specific to that dataset — so the fix
is generic: drop stopwords, de-dup, and don't truncate away the content.
"""
from core.store import fts_query


def test_content_words_survive_template_preamble():
    # 20+ words of boilerplate up front; the real content comes after.
    q = ("This comments are hidden when you submit the pull request so you do "
         "not need to remove them please fill in the following NDDataRef mask "
         "propagation regression failure")
    fq = fts_query(q).lower()
    for word in ("nddataref", "mask", "propagation", "regression"):
        assert f'"{word}"' in fq, f"content word {word!r} missing from query: {fq}"


def test_common_stopwords_dropped():
    fq = fts_query("the and or of to in is are was were this that when you")
    # a query made only of stopwords collapses to the empty match
    assert fq == '""', f"expected empty match, got {fq}"


def test_terms_deduped():
    fq = fts_query("mask mask mask regression bug")
    assert fq.count('"mask"') == 1, f"term not de-duped: {fq}"

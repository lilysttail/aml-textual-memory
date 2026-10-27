"""What did the embedding provider actually do during a ruler run?

The ruler scripts call embed_report(app.state.vc) at the end, put the dict in
their result JSON under "embedder" and print format_line(...). So a score file
always says which provider / model / dimension produced it, and a DashScope run
reports the real token count the service billed (its `usage`), not an estimate.
"""

_EMPTY = {"provider": None, "model": None, "dimension": None, "vectors_embedded": None, "tokens_used": None}


def embed_report(vc):
    """vc: the app's VectorCache (or None for a keyword-only run)."""
    emb = getattr(vc, "_emb", None) if vc is not None else None
    if emb is None:
        return dict(_EMPTY)
    return {
        "provider": getattr(emb, "provider", "unknown"),
        "model": getattr(emb, "model_name", None),
        "dimension": getattr(emb, "dim", None),
        "vectors_embedded": getattr(emb, "embedded_count", None),
        "tokens_used": getattr(emb, "tokens_used", None),   # only providers that bill by token have it
    }


def format_line(rep):
    if rep.get("provider") is None:
        return "embedder: none (keyword-only run, no vectors computed)"
    tokens = rep.get("tokens_used")
    return (f"embedder: provider={rep['provider']} model={rep['model']} dim={rep['dimension']} "
            f"vectors_embedded={rep['vectors_embedded']} "
            + ("tokens_used=n/a" if tokens is None else f"tokens_used={tokens} (as reported by the provider's usage)"))

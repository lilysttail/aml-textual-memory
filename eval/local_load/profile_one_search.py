"""Where does one search spend its time? Run INSIDE the service container (same image, same environment):

    docker exec <container> python /profile_one_search.py <space_id> "<query>" [top_k]

Builds its own core on the container's database (read path only; the query vector is computed through the
configured provider, i.e. the local stand-in), runs the search once to warm the in-memory index, then profiles
a second search and prints the top functions by cumulative time. Load-test tooling only.
"""
import cProfile
import pstats
import sys
import time

sys.path.insert(0, "/srv")          # the image's working directory

from app.config import load  # noqa: E402
from app.main import _build_embedder  # noqa: E402
from core import Core, CoreConfig  # noqa: E402
from core.db import Database  # noqa: E402


def main():
    space, query = sys.argv[1], sys.argv[2]
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    s = load()
    core = Core.build(CoreConfig(db=Database(s.db_path), paths=tuple(s.recall_paths), context_before=1, context_after=1,
                                 embedder=_build_embedder(s), embed_max_concurrency=s.embed_max_concurrency,
                                 search_heavy_concurrency=1))
    t = time.perf_counter()
    core.search(query, space_id=space, top_k=top_k)
    print(f"first search (builds the index): {time.perf_counter() - t:.3f}s")
    for name, fn in (("keyword", lambda: core.recall_candidates(query, space, limit=500, paths=("keyword",))),
                     ("dense", lambda: core.recall_candidates(query + " again", space, limit=500, paths=("dense",)))):
        t = time.perf_counter()
        fn()
        print(f"{name} path alone: {time.perf_counter() - t:.3f}s")
    prof = cProfile.Profile()
    t = time.perf_counter()
    prof.enable()
    core.search(query + " once more", space_id=space, top_k=top_k)
    prof.disable()
    print(f"second search: {time.perf_counter() - t:.3f}s")
    pstats.Stats(prof).sort_stats("cumulative").print_stats(25)


if __name__ == "__main__":
    main()

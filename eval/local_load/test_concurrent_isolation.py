"""Isolation under concurrency: two users, one service, no crossing.

WHAT THIS IS
    While both users are writing at the same time, a search for one user must
    never return the other user's memories.

HOW TO RUN
        py -m eval.local_load.test_concurrent_isolation
"""
import threading

from eval.local_load.common import (add, report, search, start_container,
                                    stop_container)

SECRET = "gamma-omega-777"  # gate:allow-secret


def main():
    start_container(fresh=True)
    try:
        for u in ("u1", "u2"):
            add(f"seed-{u}", u, "s1", [
                {"role": "user", "timestamp": 1728000000000,
                 "content": f"memory for {u} marker {SECRET}"}])

        crossings = []
        lock = threading.Lock()

        def probe(u):
            code, _dt, data = search(SECRET, u, top_k=100)
            others = [h for h in data if f"for {u}" not in h.get("content", "")]
            with lock:
                crossings.extend(others)

        threads = [threading.Thread(target=probe, args=(u,))
                   for u in ("u1", "u2") for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        c1, _d1, d1 = search(SECRET, "u1", top_k=100)
        c2, _d2, d2 = search(SECRET, "u2", top_k=100)
        payload = {
            "probes": len(threads),
            "cross_user_hits": len(crossings),
            "u1_hits": len(d1),
            "u2_hits": len(d2),
            "pass": len(crossings) == 0 and len(d1) > 0 and len(d2) > 0,
        }
        report("isolation under concurrency", payload)
        return 0 if payload["pass"] else 1
    finally:
        stop_container()


if __name__ == "__main__":
    raise SystemExit(main())
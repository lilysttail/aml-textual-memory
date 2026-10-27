"""Concurrent writes: 64 at once, like the platform does.

WHAT THIS IS
    64 threads call /add at the same time (some to the same user and session,
    some to different users), and each one searches right after its own write.
    Every write must be searchable, no 5xx may come out, and the number of
    stored rows must be exactly the number of messages written.

HOW TO RUN
        py -m eval.local_load.test_concurrent_writes
"""
from concurrent.futures import ThreadPoolExecutor

from eval.local_load.common import (add, percentiles, report, row_count,
                                    search, start_container, stop_container)

WORKERS = 64


def main():
    start_container(fresh=True)
    try:
        def one(i):
            user = "u1" if i % 2 == 0 else "u2"
            session = "s1" if i % 4 == 0 else f"s{i % 8}"
            code, dt = add(f"r{i}", user, session, [
                {"role": "user", "timestamp": 1728000000000 + i * 1000,
                 "content": f"concurrent message {i} about pool memory"}])
            scode, sdt, data = search(f"message {i}", user, top_k=100)
            return {"add_code": code, "add_s": dt, "search_code": scode,
                    "search_s": sdt, "found": bool(data)}

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            results = list(pool.map(one, range(WORKERS)))

        add_times = [r["add_s"] for r in results]
        search_times = [r["search_s"] for r in results]
        server_errors = [r for r in results
                         if r["add_code"] >= 500 or r["search_code"] >= 500]
        not_ok = [r for r in results if r["add_code"] != 200 or r["search_code"] != 200]
        not_found = [r for r in results if r["search_code"] == 200 and not r["found"]]
        rows = row_count()
        payload = {
            "workers": WORKERS,
            "add_latency_s": percentiles(add_times),
            "search_latency_s": percentiles(search_times),
            "server_errors_5xx": len(server_errors),
            "not_ok_responses": len(not_ok),
            "writes_not_searchable": len(not_found),
            "rows_in_db": rows,
            "rows_expected": WORKERS,
            "pass": (len(server_errors) == 0 and len(not_ok) == 0
                     and len(not_found) == 0 and rows == WORKERS),
        }
        report("concurrent writes (64)", payload)
        return 0 if payload["pass"] else 1
    finally:
        stop_container()


if __name__ == "__main__":
    raise SystemExit(main())
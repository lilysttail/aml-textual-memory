"""Concurrent idempotency: the same request_id, many times at once.

WHAT THIS IS
    The platform may retry an Add. The same request_id sent N times at the
    same moment must be stored exactly once, and every response must be 200.

HOW TO RUN
        py -m eval.local_load.test_concurrent_idempotent
"""
from concurrent.futures import ThreadPoolExecutor

from eval.local_load.common import (add, report, row_count, search,
                                    start_container, stop_container)

WORKERS = 32
MESSAGES = [
    {"role": "user", "timestamp": 1728000000000, "content": "the same lesson alpha"},
    {"role": "assistant", "timestamp": 1728000060000, "content": "the same reply alpha"},
]


def main():
    start_container(fresh=True)
    try:
        def one(_):
            code, _dt = add("same-request-id", "u1", "s1", MESSAGES)
            return code

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            codes = list(pool.map(one, range(WORKERS)))

        rows = row_count()
        scode, _dt, data = search("alpha", "u1", top_k=100)
        payload = {
            "workers": WORKERS,
            "requests": len(codes),
            "not_200": len([c for c in codes if c != 200]),
            "rows_in_db": rows,
            "rows_expected": len(MESSAGES),
            "search_code": scode,
            "hits": len(data),
            "hits_expected": len(MESSAGES),
            "pass": (all(c == 200 for c in codes) and rows == len(MESSAGES)
                     and scode == 200 and len(data) == len(MESSAGES)),
        }
        report("concurrent idempotency", payload)
        return 0 if payload["pass"] else 1
    finally:
        stop_container()


if __name__ == "__main__":
    raise SystemExit(main())
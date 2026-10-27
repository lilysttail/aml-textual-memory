"""Restart: the data must survive, and the start-up check must not refuse it.

WHAT THIS IS
    The service stores its data in a Docker volume (like Zeabur's /data).
    We write, stop the container, start it again, and check that the rows are
    still there, still searchable, and that the write-time settings check
    (core_write_settings) does not stop the start.

HOW TO RUN
        py -m eval.local_load.test_restart_persistence
"""
from eval.local_load.common import (CONTAINER, add, docker, report, row_count,
                                    search, start_container, stop_container,
                                    wait_healthy)


def main():
    start_container(fresh=True)
    try:
        code, _dt = add("r1", "u1", "s1", [
            {"role": "user", "timestamp": 1728000000000, "content": "survives a restart alpha"},
            {"role": "assistant", "timestamp": 1728000060000, "content": "and this one too alpha"}])
        before_rows = row_count()
        scode, _dt, before_data = search("alpha", "u1", top_k=100)

        docker("restart", CONTAINER)
        wait_healthy()

        after_rows = row_count()
        scode2, _dt2, after_data = search("alpha", "u1", top_k=100)
        logs = docker("logs", "--tail", "50", CONTAINER)
        refused = "refusing to start" in logs.lower()
        payload = {
            "add_code": code,
            "rows_before": before_rows,
            "rows_after": after_rows,
            "hits_before": len(before_data),
            "hits_after": len(after_data),
            "search_code_after": scode2,
            "start_up_check_refused": refused,
            "pass": (before_rows == after_rows and len(after_data) > 0
                     and scode2 == 200 and not refused),
        }
        report("restart keeps the data", payload)
        return 0 if payload["pass"] else 1
    finally:
        stop_container()


if __name__ == "__main__":
    raise SystemExit(main())
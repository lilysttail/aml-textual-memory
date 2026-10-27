"""Bad input: 4xx, never 5xx. And all three key schemes are accepted.

WHAT THIS IS
    The service is asked for things it must refuse (broken JSON, missing
    fields, a wrong key) and for things it must accept (the three key
    schemes: Bearer, Token, X-Api-Key). A refusal must be a 4xx; a 5xx would
    look like our bug to the platform.

HOW TO RUN
        py -m eval.local_load.test_bad_input
"""
import httpx

from eval.local_load.common import (API_KEY, AUTH, BASE, add, report,
                                    start_container, stop_container)


def main():
    start_container(fresh=True)
    try:
        results = {}

        # a broken body
        r = httpx.post(f"{BASE}/add", content=b"{not json", headers=AUTH, timeout=30)
        results["broken_json"] = r.status_code

        # missing fields
        r = httpx.post(f"{BASE}/add", json={"request_id": "x"}, headers=AUTH, timeout=30)
        results["missing_fields"] = r.status_code
        r = httpx.post(f"{BASE}/add", json={"request_id": "x", "user_id": "u",
                                            "session_id": "s", "messages": []},
                       headers=AUTH, timeout=30)
        results["empty_messages"] = r.status_code
        r = httpx.post(f"{BASE}/search", json={"query": "x", "user_id": "u",
                                               "top_k": 0}, headers=AUTH, timeout=30)
        results["top_k_zero"] = r.status_code

        # a very long content: the official contract sets no length limit, so
        # this is what we do with it (recorded, not judged here)
        long_text = "x" * (1024 * 1024)
        code, _dt = add("long", "u1", "s1", [
            {"role": "user", "timestamp": 1728000000000, "content": long_text}])
        results["one_mb_content"] = code

        # the wrong key
        r = httpx.post(f"{BASE}/add", json={"request_id": "y", "user_id": "u",
                                            "session_id": "s", "messages": []},
                       headers={"Authorization": "Bearer wrong-key"}, timeout=30)
        results["wrong_key"] = r.status_code
        r = httpx.get(f"{BASE}/health", timeout=30)
        results["health_without_key"] = r.status_code

        # the three key schemes
        body = {"request_id": "k1", "user_id": "u1", "session_id": "s1",
                "messages": [{"role": "user", "timestamp": 1728000000000,
                              "content": "key scheme alpha"}]}
        schemes = {
            "bearer": {"Authorization": f"Bearer {API_KEY}"},
            "token": {"Authorization": f"Token {API_KEY}"},
            "x_api_key": {"X-Api-Key": API_KEY},
        }
        for name, h in schemes.items():
            r = httpx.post(f"{BASE}/add", json=body, headers=h, timeout=30)
            results[f"key_{name}"] = r.status_code

        refused = {k: v for k, v in results.items() if v >= 500}
        payload = {
            "statuses": results,
            "server_errors_5xx": len(refused),
            # the exact statuses: the contract says an input we cannot use is
            # refused (4xx), and a legal long message is taken (the official
            # contract sets no length limit — 柯 10-21 判：保持)
            "pass": len(refused) == 0
                    and results["broken_json"] == 422
                    and results["missing_fields"] == 422
                    and results["empty_messages"] == 422
                    and results["top_k_zero"] == 422
                    and results["one_mb_content"] == 200
                    and results["wrong_key"] == 401
                    and results["health_without_key"] == 200
                    and results["key_bearer"] == 200
                    and results["key_token"] == 200
                    and results["key_x_api_key"] == 200,
        }
        report("bad input and key schemes", payload)
        return 0 if payload["pass"] else 1
    finally:
        stop_container()


if __name__ == "__main__":
    raise SystemExit(main())
"""Shared plumbing for the local load tests.

WHAT THIS IS
    The service under test runs in a Docker container on this machine
    (built from this branch). These scripts are the CLIENT side: they drive
    the container over HTTP on 127.0.0.1, exactly like the platform would.

HOW TO RUN
    From the repository root, with the project's own Python:
        py -m eval.local_load.test_concurrent_writes
    Every script can be run on its own. They all need:
      - Docker inside WSL Ubuntu (the scripts call it through
        `wsl -d Ubuntu -e docker ...`);
      - an image built from this branch:
            wsl -d Ubuntu -e docker build -t aml-local-test:local .
      - a free port (AML_PORT, default 18080).

    The container gets a throwaway API key (local only) and a Docker volume
    for /data, so the database survives a restart of the container.

    Nothing here touches a real model, a real key or the evaluation data.
"""
import json
import os
import statistics
import subprocess
import time

import httpx

WSL = ["wsl", "-d", "Ubuntu", "-e"]
WSL_ROOT = ["wsl", "-d", "Ubuntu", "-u", "root", "-e"]
IMAGE = os.environ.get("AML_IMAGE", "aml-local-test:local")
CONTAINER = os.environ.get("AML_CONTAINER", "aml-local-test")
PORT = int(os.environ.get("AML_PORT", "18080"))
API_KEY = os.environ.get("AML_KEY", "local-test-key")
DATA_VOLUME = os.environ.get("AML_DATA_VOLUME", "/tmp/aml-local-data")
BASE = f"http://127.0.0.1:{PORT}"
AUTH = {"Authorization": f"Bearer {API_KEY}"}


def docker(*args, check=True, timeout=900):
    """Run a docker command inside WSL Ubuntu."""
    r = subprocess.run(WSL + ["docker"] + list(args), capture_output=True,
                       text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args)} failed:\n{r.stderr[-3000:]}")
    return r.stdout.strip()


def inside_health(name=None):
    """Ask the service from INSIDE the container (is the service up at all?)."""
    code = ("import urllib.request;"
            "print(urllib.request.urlopen('http://127.0.0.1:8080/health',"
            "timeout=5).read().decode())")
    return docker("exec", name or CONTAINER, "python", "-c", code, check=False)


def wait_healthy(timeout=90.0):
    """Wait until /health answers 200 from this machine."""
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            r = httpx.get(f"{BASE}/health", timeout=10.0, trust_env=False)
            if r.status_code == 200:
                return True
            last = f"{r.status_code} {r.text[:200]}"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(1.0)
    inside = inside_health()
    raise RuntimeError(f"the service never became healthy from this machine: "
                       f"{last} | from inside the container: {inside[-300:]}")


def start_container(fresh=True, image=None, env=None, volume=True, name=None,
                clear_data=True):
    """Start the service container and wait for /health."""
    name = name or CONTAINER
    if fresh:
        docker("rm", "-f", name, check=False)
    if fresh and clear_data:
        # A fresh container with the OLD data directory would still hold the
        # previous run's rows (the mount is a host path, not a named volume, so
        # removing the container does not remove it). Clear it.
        # the directory belongs to root (docker created it), so clear it as root
        subprocess.run(WSL_ROOT + ["rm", "-rf", DATA_VOLUME], capture_output=True)
    if volume:
        docker("volume", "create", DATA_VOLUME, check=False)
    args = ["run", "-d", "--name", name, "-p", f"{PORT}:8080"]
    if volume:
        args += ["-v", f"{DATA_VOLUME}:/data"]
    args += ["-e", f"API_KEY={API_KEY}", "-e", "DB_PATH=/data/memory.db"]
    for k, v in (env or {}).items():
        args += ["-e", f"{k}={v}"]
    args += [image or IMAGE]
    docker(*args)
    wait_healthy()
    return name


def stop_container(name=None):
    docker("rm", "-f", name or CONTAINER, check=False)


def add(request_id, user_id, session_id, messages, timeout=120.0, headers=None):
    """POST /add. Returns (status_code, seconds)."""
    body = {"request_id": request_id, "user_id": user_id,
            "session_id": session_id, "messages": messages}
    t0 = time.perf_counter()
    r = httpx.post(f"{BASE}/add", json=body, headers=headers or AUTH,
                   timeout=timeout, trust_env=False)
    return r.status_code, time.perf_counter() - t0


def search(query, user_id, top_k=100, timeout=120.0, headers=None, options=None):
    """POST /search. Returns (status_code, seconds, data)."""
    body = {"query": query, "user_id": user_id, "top_k": top_k}
    if options is not None:
        body["options"] = options
    t0 = time.perf_counter()
    r = httpx.post(f"{BASE}/search", json=body, headers=headers or AUTH,
                   timeout=timeout, trust_env=False)
    dt = time.perf_counter() - t0
    data = []
    if r.status_code == 200:
        try:
            data = r.json().get("data", [])
        except Exception:
            data = []
    return r.status_code, dt, data


def row_count(name=None):
    """How many rows the service stored (read inside the container)."""
    code = ("import sqlite3;c=sqlite3.connect('/data/memory.db');"
            "print(c.execute('select count(*) from core_records').fetchone()[0])")
    out = docker("exec", name or CONTAINER, "python", "-c", code)
    return int(out.strip().splitlines()[-1])


def percentiles(values):
    """median / p95 / max / mean of a list of durations."""
    if not values:
        return {"n": 0}
    s = sorted(values)

    def pct(p):
        if len(s) == 1:
            return s[0]
        k = (len(s) - 1) * p
        lo = int(k)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    return {"n": len(s), "median": round(pct(0.5), 4), "p95": round(pct(0.95), 4),
            "max": round(s[-1], 4), "mean": round(statistics.fmean(s), 4)}


def parse_memory(text):
    """Read the memory USAGE out of a docker stats line into bytes.

    docker stats prints "used / limit" (e.g. "55.5MiB / 4GiB"). Only the
    left side is the usage; the right side is the limit and must never be
    read as a usage. GiB is 1024^3 bytes, MiB 1024^2, KiB 1024, B 1.
    """
    left = str(text).split("/")[0]
    for part in left.split():
        if part.endswith("GiB"):
            return int(float(part[:-3]) * 1024 ** 3)
        if part.endswith("MiB"):
            return int(float(part[:-3]) * 1024 ** 2)
        if part.endswith("KiB"):
            return int(float(part[:-3]) * 1024)
        if part.endswith("B"):
            return int(float(part[:-1]))
    return 0

def report(name, payload):
    """Print one result block."""
    print(f"=== {name} ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print()
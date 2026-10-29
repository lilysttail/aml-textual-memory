"""Search-phase load test under the conditions of the deployment: 2 CPUs, a hard memory limit, the meaning-based path ON,
a database the size of a real evaluation, and 32 simultaneous searches aimed at the biggest spaces.

Why it exists: the earlier full-corpus test ran 8 searches at a time, with no CPU or memory limit, on ten public
conversations, and with the meaning-based path OFF (the container only got API_KEY / DB_PATH) — so it never saw what an
evaluation does to a 2-core host. This one starts the service exactly as deployed (same switches) and points the
embedding adapter at a local stand-in (eval/local_load/fake_embed.py) through DASHSCOPE_BASE_URL, so the real adapter,
gate and deadlines run; only the far end is local. No real model, no real key, no evaluation data.

Docker runs inside WSL Ubuntu (like the other local load tests). Two containers on one docker network:
    <prefix>-fake   the stand-in embedding service (same image, python + numpy), port 8099 inside the network
    <prefix>-svc    the service under test, --cpus / --memory as given, DASHSCOPE_BASE_URL=http://<prefix>-fake:8099/v1
The service's data directory is a WSL path (not a named volume), so `up --keep-data` reuses a loaded database.
While `load` / `query` run they hold one process open inside the distro: WSL otherwise stops an idle distro (dockerd and
the containers with it) a few seconds after the last `wsl` command ends, and HTTP traffic does not count as activity.

    wsl -d Ubuntu -e docker build -t aml462:base-<commit> .          # an image of the version under test
    py -m eval.local_load.search_overload up    --image aml462:base-<commit> --cpus 2 --memory 1.5g [--keep-data]
    py -m eval.local_load.search_overload load  --out <dir>           # writes the synthetic data through /add (16 at a time)
    py -m eval.local_load.search_overload query --out <dir> --mode largest|top10|small [--queries 128] [--concurrency 32]
    py -m eval.local_load.search_overload down

Data (fixed seed, SEED below): 1433 spaces; sizes heavy-tailed with the largest 9806 and a mean of 301 (about 431k
records); texts drawn from a pool of 136k distinct sentences, so about 130k distinct contents (the embedding
store keeps one vector per distinct text). `load` writes data_stats.json with the counts it actually wrote.

Every `query` round records: per-search seconds (p50 / p95 / max), non-200 answers (503 separately, client timeouts
separately), how many searches dropped their meaning-based half (counted from the service log), the container's memory
peak (cgroup v2 memory.current, and anon / file / shmem from memory.stat, sampled every 0.1 s inside the container; plus
memory.peak, marked as not this round's where the kernel can't reset it), whether the kernel killed
it (OOM), how long each /health took while the round ran, and how many calls the stand-in embedding service answered.
Results go to <out>/round_<mode>_<time>.json; nothing about the content of any text is written.
"""
import argparse
import contextlib
import json
import math
import os
import random
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

WSL = ["wsl", "-d", "Ubuntu", "-e"]
WSL_ROOT = ["wsl", "-d", "Ubuntu", "-u", "root", "-e"]
SERVICE_KEY = "load-test-service-key-local-only"       # the service's own key for this test (REQUIRE_REAL_API_KEY=1 needs a non-placeholder)
FAKE_EMBED_KEY = "fake-embed-key-local-only"            # the adapter sends it; the stand-in ignores it
HERE = os.path.dirname(os.path.abspath(__file__))
SKIP_LINE = "dense query vector skipped"
SEED = 462


def docker(*args, check=True, timeout=1800):
    r = subprocess.run(WSL + ["docker"] + list(args), capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:3])} ... failed:\n{r.stderr[-2000:]}")
    return r.stdout.strip()


@contextlib.contextmanager
def wsl_kept_awake():
    """Hold one process open inside the distro for the whole run.

    WSL stops an idle distro (and with it dockerd and the containers) a few seconds
    after the last `wsl` process exits; the HTTP traffic from this machine does not
    count as activity. Without this a long load is cut off soon after it starts.
    """
    keeper = subprocess.Popen(WSL + ["sleep", "infinity"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield
    finally:
        keeper.terminate()


def container_state(n):
    return docker("inspect", "-f", "{{.State.Status}} oomkilled={{.State.OOMKilled}} restarts={{.RestartCount}} "
                  "started={{.State.StartedAt}}", n.svc, check=False)


def wsl_path(win_path):
    p = os.path.abspath(win_path).replace("\\", "/")
    return f"/mnt/{p[0].lower()}{p[2:]}"


class Names:
    def __init__(self, prefix, port, fake_port):
        self.prefix = prefix
        self.net = f"{prefix}-net"
        self.svc = f"{prefix}-svc"
        self.fake = f"{prefix}-fake"
        # Not /tmp: in WSL /tmp is a RAM disk, and a database there is charged to the container's memory
        # (as shmem) — the deployment keeps it on a disk. /var/tmp is on the distro's ext4 disk.
        self.data = f"/var/tmp/{prefix}-data"
        self.base = f"http://127.0.0.1:{port}"
        self.fake_base = f"http://127.0.0.1:{fake_port}"
        self.port = port
        self.fake_port = fake_port


def up(n, image, cpus, memory, delay, keep_data, extra_env=()):
    for c in (n.svc, n.fake):
        docker("rm", "-f", c, check=False)
    docker("network", "create", n.net, check=False)
    if not keep_data:
        subprocess.run(WSL_ROOT + ["rm", "-rf", n.data], capture_output=True)
    subprocess.run(WSL + ["mkdir", "-p", n.data], capture_output=True)
    # the stand-in: same image (python + numpy), the script mounted read-only; no limits (it plays a remote service)
    docker("run", "-d", "--name", n.fake, "--network", n.net, "-p", f"{n.fake_port}:8099",
           "-v", f"{wsl_path(os.path.join(HERE, 'fake_embed.py'))}:/fake_embed.py:ro",
           "--entrypoint", "python", image, "/fake_embed.py", "--port", "8099", "--delay", str(delay))
    env = {"RECALL_PATHS": "keyword,dense", "EMBED_PROVIDER": "dashscope", "RERANK_ENABLED": "0", "STEM_ENGLISH": "0",
           "NEIGHBOUR_CANDIDATES": "1", "EMBED_MAX_CONCURRENCY": "64", "DASHSCOPE_DIM": "1024", "REQUIRE_REAL_API_KEY": "1",
           "API_KEY": SERVICE_KEY, "DASHSCOPE_API_KEY": FAKE_EMBED_KEY, "DASHSCOPE_BASE_URL": f"http://{n.fake}:8099/v1",  # gate:allow-id
           "DB_PATH": "/data/memory.db"}
    env.update(dict(extra_env))                       # --env KEY=VALUE: only for runs that say so (e.g. the result comparison)
    args = ["run", "-d", "--name", n.svc, "--network", n.net, "-p", f"{n.port}:8080", "--cpus", str(cpus), "--memory", memory,
            "-v", f"{n.data}:/data"]
    for k, v in env.items():
        args += ["-e", f"{k}={v}"]
    docker(*args, image)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            if httpx.get(n.base + "/health", timeout=10, trust_env=False).status_code == 200:
                limits = docker("exec", n.svc, "sh", "-c", "cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max")
                print(json.dumps({"up": True, "image": image, "cgroup_cpu_max_and_memory_max": limits.split("\n")}))
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("service did not come up: " + docker("logs", "--tail", "40", n.svc, check=False))


def down(n):
    for c in (n.svc, n.fake):
        docker("rm", "-f", c, check=False)


# ---- synthetic data (fixed seed) ----

def space_sizes(rng, n_spaces=1433, largest=9806, mean=301, top=(9806, 7400, 6100, 5200, 4600, 4100, 3700, 3300, 3000, 2700)):
    """Heavy-tailed sizes: the ten biggest fixed (largest first), the rest lognormal, scaled so the mean is `mean`."""
    total = n_spaces * mean
    rest_n = n_spaces - len(top)
    rest_total = total - sum(top)
    raw = [rng.lognormvariate(0, 1.1) for _ in range(rest_n)]
    scale = rest_total / sum(raw)
    rest = [max(1, min(largest - 1, int(round(x * scale)))) for x in raw]
    diff = rest_total - sum(rest)          # spread the rounding remainder one record at a time
    i = 0
    while diff != 0:
        j = i % rest_n
        if diff > 0 and rest[j] < 2600:
            rest[j] += 1
            diff -= 1
        elif diff < 0 and rest[j] > 1:
            rest[j] -= 1
            diff += 1
        i += 1
    return list(top) + rest


WORDS = None


def vocabulary(rng, n=6000):
    syl = ["ka", "ri", "mo", "ten", "sa", "lu", "ve", "dor", "pi", "an", "el", "ost", "ma", "ne", "ul", "ra", "to", "ki", "ber", "sol"]
    out = set()
    while len(out) < n:
        out.add("".join(rng.choice(syl) for _ in range(rng.randint(2, 4))))
    return sorted(out)


def sentence_pool(rng, words, n=136000):
    pool, seen = [], set()
    while len(pool) < n:
        s = " ".join(rng.choice(words) for _ in range(rng.randint(8, 20)))
        if s not in seen:
            seen.add(s)
            pool.append(s)
    return pool


def build_data(seed=SEED):
    rng = random.Random(seed)
    words = vocabulary(rng)
    pool = sentence_pool(rng, words)
    sizes = space_sizes(rng)
    spaces = []
    for s, size in enumerate(sizes):
        spaces.append({"user_id": f"load-u{s:04d}", "size": size, "picks": [rng.randrange(len(pool)) for _ in range(size)]})
    return pool, spaces


def batches_of(space, pool, per_batch=20, t0=1_700_000_000_000):
    """One /add per 20 messages, sessions of 40 messages, times one minute apart."""
    out = []
    picks = space["picks"]
    for b in range(0, len(picks), per_batch):
        msgs = [{"role": "user" if (b + i) % 2 == 0 else "assistant", "timestamp": t0 + (b + i) * 60_000, "content": pool[p]}
                for i, p in enumerate(picks[b:b + per_batch])]
        out.append({"request_id": f"{space['user_id']}-b{b // per_batch:05d}", "user_id": space["user_id"],
                    "session_id": f"{space['user_id']}-s{b // 40:05d}", "messages": msgs})
    return out


def load(n, out_dir, concurrency=16):
    os.makedirs(out_dir, exist_ok=True)
    pool, spaces = build_data()
    work = [b for sp in spaces for b in batches_of(sp, pool)]
    auth = {"Authorization": f"Bearer {SERVICE_KEY}"}
    state_before = container_state(n)
    lock = threading.Lock()
    done = {"ok": 0, "bad": 0, "codes": {}}
    started = time.time()
    with httpx.Client(timeout=600, trust_env=False) as client:
        def one(body):
            for attempt in range(5):
                try:
                    r = client.post(n.base + "/add", json=body, headers=auth)
                    code = r.status_code
                except Exception as e:
                    code = type(e).__name__
                if code == 200:
                    break
                time.sleep(2 * (attempt + 1))
            with lock:
                done["ok" if code == 200 else "bad"] += 1
                done["codes"][str(code)] = done["codes"].get(str(code), 0) + 1
                k = done["ok"] + done["bad"]
                if k % 2000 == 0:
                    print(f"  {k}/{len(work)} batches, {time.time() - started:.0f}s", flush=True)
        with ThreadPoolExecutor(concurrency) as ex:
            list(ex.map(one, work))
    distinct = len({p for sp in spaces for p in sp["picks"]})
    sizes = sorted((sp["size"] for sp in spaces), reverse=True)
    stats = {"seed": SEED, "spaces": len(spaces), "records": sum(sizes), "largest": sizes[0], "mean": round(sum(sizes) / len(sizes), 1),
             "top10": sizes[:10], "distinct_contents": distinct, "batches": len(work), "batches_ok": done["ok"], "batch_codes": done["codes"],
             "add_concurrency": concurrency, "seconds": round(time.time() - started, 1), "fake_embed_stats": stats_of(n),
             "container_state_before": state_before, "container_state_after": container_state(n)}
    # A load that lost batches, or whose container was restarted under it, is not the data the rounds assume.
    stats["complete"] = done["bad"] == 0 and stats["container_state_before"] == stats["container_state_after"]
    with open(os.path.join(out_dir, "data_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


def stats_of(n):
    try:
        return httpx.get(n.fake_base + "/stats", timeout=10, trust_env=False).json()
    except Exception as e:
        return {"error": type(e).__name__}


# ---- one query round ----

def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1))
    return round(xs[k], 3)


def query(n, out_dir, mode, n_queries=128, concurrency=32, top_k=100, client_timeout=300.0, seed=SEED,
          save_answers=False):
    pool, spaces = build_data()
    by_size = sorted(spaces, key=lambda s: -s["size"])
    rng = random.Random(seed * 7 + {"largest": 1, "top10": 2, "small": 3, "all": 4}[mode])
    if mode == "largest":
        targets = [by_size[0]] * n_queries
    elif mode == "top10":
        targets = [by_size[i % 10] for i in range(n_queries)]
    elif mode == "all":                                              # every space in turn: fills the memory budget
        order = list(spaces)
        rng.shuffle(order)
        targets = [order[i % len(order)] for i in range(n_queries)]
    else:                                                            # the control: spread over ordinary spaces
        small = [s for s in spaces if s["size"] <= 400]
        targets = [rng.choice(small) for _ in range(n_queries)]
    queries = []
    for t in targets:
        words = pool[rng.choice(t["picks"])].split()
        k = rng.randint(3, 5)
        i = rng.randrange(max(1, len(words) - k))
        queries.append({"query": " ".join(words[i:i + k]), "user_id": t["user_id"], "top_k": top_k})

    auth = {"Authorization": f"Bearer {SERVICE_KEY}"}
    state_before = container_state(n)
    # memory.peak can only be reset on newer kernels (writing to it); where it can't, it still holds the load phase's
    # peak, so it is reported but marked as not belonging to this round. The per-round numbers are the samples.
    docker("exec", n.svc, "sh", "-c", "echo 0 > /sys/fs/cgroup/memory.peak", check=False)
    peak_before = read_cg(n, "memory.peak")
    current_before = read_cg(n, "memory.current")
    peak_is_this_round = isinstance(peak_before, int) and isinstance(current_before, int) and peak_before <= current_before + 64 * 2 ** 20
    oom_before = read_oom(n)
    fake_before = stats_of(n)
    skips_before = count_skips(n)          # the whole log, before and after: the difference is this round's
    # every 0.1 s: memory.current, then anon / file / shmem from memory.stat (anon = the process's own memory;
    # file = page cache, which the kernel gives back under pressure)
    sampler = subprocess.Popen(WSL + ["docker", "exec", n.svc, "sh", "-c",
                                      "while true; do echo $(cat /sys/fs/cgroup/memory.current) "
                                      "$(grep -E '^(anon|file|shmem) ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); sleep 0.1; done"],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    samples = []

    def read_samples():
        for line in sampler.stdout:
            parts = line.split()
            if len(parts) == 4 and all(p.isdigit() for p in parts):
                samples.append(tuple(int(p) for p in parts))
    threading.Thread(target=read_samples, daemon=True).start()
    health = []
    stop = threading.Event()

    def poll_health():
        while not stop.is_set():
            t = time.time()
            try:
                code = httpx.get(n.base + "/health", timeout=client_timeout, trust_env=False).status_code
            except Exception as e:
                code = type(e).__name__
            health.append({"seconds": round(time.time() - t, 3), "code": code})
            stop.wait(3)
    hp = threading.Thread(target=poll_health, daemon=True)
    hp.start()

    results = []
    answers = [None] * len(queries)                   # per query (by position): ids and scores, for --save-answers
    lock = threading.Lock()
    started = time.time()
    with httpx.Client(timeout=client_timeout, trust_env=False, limits=httpx.Limits(max_connections=concurrency + 4)) as client:
        def one(iq):
            i, q = iq
            t = time.time()
            data = None
            try:
                r = client.post(n.base + "/search", json=q, headers=auth)
                code = r.status_code
                if code == 200:
                    data = r.json().get("data", [])
                hits = len(data) if data is not None else None
            except httpx.TimeoutException:
                code, hits = "client_timeout", None
            except Exception as e:
                code, hits = type(e).__name__, None
            with lock:
                results.append({"seconds": round(time.time() - t, 3), "code": code, "hits": hits})
                answers[i] = {"code": code, "ids": [d["id"] for d in data] if data is not None else None,
                              "scores": [d["score"] for d in data] if data is not None else None}
        with ThreadPoolExecutor(concurrency) as ex:
            list(ex.map(one, enumerate(queries)))
    wall = time.time() - started
    stop.set()
    hp.join(timeout=client_timeout + 5)
    sampler.terminate()

    lat = [r["seconds"] for r in results if r["code"] == 200]
    codes = {}
    for r in results:
        codes[str(r["code"])] = codes.get(str(r["code"]), 0) + 1
    skips = count_skips(n) - skips_before
    fake_after = stats_of(n)
    state = container_state(n)
    rep = {
        "mode": mode, "query_seed": seed, "queries": len(queries), "concurrency": concurrency, "top_k": top_k, "wall_seconds": round(wall, 1),
        "targets": {"distinct_spaces": len({q["user_id"] for q in queries}),
                    "sizes": sorted({t["size"] for t in targets}, reverse=True)[:10]},
        "status_counts": codes,
        "non_200": sum(v for k, v in codes.items() if k != "200"),
        "status_503": codes.get("503", 0),
        "client_timeouts": codes.get("client_timeout", 0),
        "latency_200_seconds": {"p50": pct(lat, 0.5), "p95": pct(lat, 0.95), "max": max(lat) if lat else None, "n": len(lat)},
        "all_latency_seconds_max": max(r["seconds"] for r in results) if results else None,
        "dense_skipped_log_lines": skips,
        # one embedding call per search that got its query vector (all stored vectors exist after the load)
        "embed_calls_this_round": (fake_after.get("calls", 0) - fake_before.get("calls", 0)) if "calls" in fake_after else None,
        "memory": {"limit": read_cg(n, "memory.max"),
                   "peak_bytes": read_cg(n, "memory.peak"), "peak_is_this_round": peak_is_this_round,
                   "peak_bytes_before": peak_before, "current_bytes_before": current_before,
                   "sampled_current_max_bytes": max((x[0] for x in samples), default=None),
                   "sampled_anon_max_bytes": max((x[1] for x in samples), default=None),
                   "sampled_file_max_bytes": max((x[2] for x in samples), default=None),
                   "sampled_shmem_max_bytes": max((x[3] for x in samples), default=None),
                   "samples": len(samples),
                   "oom_kill_events_before": oom_before, "oom_kill_events_after": read_oom(n)},
        "container_state_before": state_before,
        "container_state_after": state,
        "health": {"n": len(health), "max_seconds": max((h["seconds"] for h in health), default=None),
                   "p50_seconds": pct([h["seconds"] for h in health], 0.5), "codes": sorted({str(h["code"]) for h in health}),
                   "each": health},
        "fake_embed_calls": {"before": fake_before, "after": fake_after},
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"round_{mode}_{time.strftime('%Y%m%dT%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    brief = {k: rep[k] for k in ("mode", "wall_seconds", "status_counts", "latency_200_seconds", "all_latency_seconds_max",
                                  "dense_skipped_log_lines", "embed_calls_this_round", "container_state_after")}
    mib = lambda b: round(b / 2 ** 20, 1) if isinstance(b, int) else b
    m = rep["memory"]
    brief["memory_mib"] = {"limit": mib(m["limit"]), "peak": mib(m["peak_bytes"]), "peak_is_this_round": m["peak_is_this_round"],
                           "sampled_current_max": mib(m["sampled_current_max_bytes"]), "sampled_anon_max": mib(m["sampled_anon_max_bytes"]),
                           "sampled_file_max": mib(m["sampled_file_max_bytes"]), "sampled_shmem_max": mib(m["sampled_shmem_max_bytes"]),
                           "oom_kills": (m["oom_kill_events_after"] or 0) - (m["oom_kill_events_before"] or 0)}
    brief["health"] = {k: rep["health"][k] for k in ("n", "max_seconds", "p50_seconds", "codes")}
    brief["file"] = path
    if save_answers:                                  # ids and scores only, no text
        apath = os.path.join(out_dir, f"answers_{mode}_{time.strftime('%Y%m%dT%H%M%S')}.json")
        with open(apath, "w", encoding="utf-8") as f:
            json.dump({"mode": mode, "seed": seed, "top_k": top_k, "answers": answers}, f)
        brief["answers_file"] = apath
    print(json.dumps(brief, indent=1))


def compare(path_a, path_b, tolerance=1e-5):
    """Two answer files of the same queries: are the id lists identical, and how far apart are the scores?"""
    a = json.load(open(path_a, encoding="utf-8"))
    b = json.load(open(path_b, encoding="utf-8"))
    assert (a["mode"], a["seed"], a["top_k"]) == (b["mode"], b["seed"], b["top_k"]), "not the same queries"
    same_ids, diff_ids, not_200, worst = 0, [], [], 0.0
    for i, (x, y) in enumerate(zip(a["answers"], b["answers"])):
        if x is None or y is None or x["code"] != 200 or y["code"] != 200:
            not_200.append(i)
            continue
        if x["ids"] == y["ids"]:
            same_ids += 1
            worst = max([worst] + [abs(p - q) for p, q in zip(x["scores"], y["scores"])])
        else:
            first = next((k for k, (p, q) in enumerate(zip(x["ids"], y["ids"])) if p != q), min(len(x["ids"]), len(y["ids"])))
            diff_ids.append({"query": i, "first_difference_at": first, "len_a": len(x["ids"]), "len_b": len(y["ids"]),
                             "same_set": sorted(x["ids"]) == sorted(y["ids"])})
    rep = {"queries": len(a["answers"]), "identical_id_lists": same_ids, "different_id_lists": len(diff_ids),
           "not_200": not_200, "max_score_difference_on_identical_lists": worst, "within_tolerance": worst <= tolerance,
           "differences": diff_ids, "a": path_a, "b": path_b}
    print(json.dumps(rep, indent=1))
    return rep


def count_skips(n):
    """How many times the service has logged that a search dropped its meaning-based half (whole log so far)."""
    r = subprocess.run(WSL + ["docker", "logs", n.svc], capture_output=True, text=True, errors="replace")
    return (r.stdout + r.stderr).count(SKIP_LINE)


def read_cg(n, name):
    v = docker("exec", n.svc, "cat", f"/sys/fs/cgroup/{name}", check=False)
    return int(v) if v.strip().isdigit() else (v or None)


def read_oom(n):
    v = docker("exec", n.svc, "sh", "-c", "grep -E '^oom_kill ' /sys/fs/cgroup/memory.events | cut -d' ' -f2", check=False)
    return int(v) if v.strip().isdigit() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["up", "load", "query", "down", "data-stats", "compare"])
    ap.add_argument("--prefix", default="aml462")
    ap.add_argument("--port", type=int, default=18462)
    ap.add_argument("--fake-port", type=int, default=18463)
    ap.add_argument("--image", default="aml462:base")
    ap.add_argument("--cpus", default="2")
    ap.add_argument("--memory", default="1.5g")
    ap.add_argument("--delay", type=float, default=0.5, help="the stand-in embedding service's fixed delay per call (seconds)")
    ap.add_argument("--keep-data", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=["largest", "top10", "small", "all"], default="largest",
                    help="all: every space in turn (use --queries 1433 to reach each once) — the memory-budget round")
    ap.add_argument("--queries", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--client-timeout", type=float, default=300.0)
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="extra environment for the service container (up); repeatable")
    ap.add_argument("--save-answers", action="store_true", help="query: also write each search's ids and scores")
    ap.add_argument("--query-seed", type=int, default=SEED,
                    help="query: seed of the query selection (same spaces, other queries — so their vectors are new)")
    ap.add_argument("--a")
    ap.add_argument("--b")
    args = ap.parse_args(argv)
    n = Names(args.prefix, args.port, args.fake_port)
    if args.cmd == "up":
        up(n, args.image, args.cpus, args.memory, args.delay, args.keep_data,
           extra_env=[tuple(e.split("=", 1)) for e in args.env])
    elif args.cmd == "down":
        down(n)
    elif args.cmd == "load":
        with wsl_kept_awake():
            load(n, args.out)
    elif args.cmd == "query":
        with wsl_kept_awake():
            query(n, args.out, args.mode, args.queries, args.concurrency, client_timeout=args.client_timeout,
                  seed=args.query_seed, save_answers=args.save_answers)
    elif args.cmd == "compare":
        compare(args.a, args.b)
    else:
        pool, spaces = build_data()
        sizes = sorted((s["size"] for s in spaces), reverse=True)
        print(json.dumps({"spaces": len(spaces), "records": sum(sizes), "largest": sizes[0], "mean": round(sum(sizes) / len(sizes), 1),
                          "top10": sizes[:10], "distinct_contents": len({p for s in spaces for p in s["picks"]}), "pool": len(pool)}))


if __name__ == "__main__":
    main()

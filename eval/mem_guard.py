"""Memory guard that LAUNCHES the bench as a child and watches it by PID.

The old version matched processes by command line containing a string, which
also matched the guard's own argv -> it watched itself. This version launches
the bench command as a subprocess and monitors exactly that process tree
(target pid + its children), never the guard's own pid.

Every poll: read system available RAM and sum the tree's current working set +
commit (pagefile), tracking the peak. If system available RAM drops below the
threshold, kill the whole tree and report.

Usage:
  python eval/mem_guard.py <threshold_gb> -- <command to run the bench ...>
e.g.
  python eval/mem_guard.py 8 -- ..\\.venv_scaffold\\Scripts\\python.exe eval\\bench_concurrent.py 300 32 8
"""
import os
import subprocess
import sys
import time

import psutil


def run_guarded(argv, threshold_gb=8.0, poll=1.0, max_s=1800.0):
    me = os.getpid()
    proc = subprocess.Popen(argv)
    target = psutil.Process(proc.pid)
    print(f"GUARD up: watching pid={proc.pid} (excluding self={me}) "
          f"threshold={threshold_gb}GB", flush=True)

    t0 = time.time()
    min_avail = float("inf")
    peak_ws = 0
    peak_commit = 0
    killed = False
    tick = 0

    def tree():
        procs = []
        try:
            procs = [target] + target.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
        return [p for p in procs if p.pid != me]

    while proc.poll() is None and time.time() - t0 < max_s:
        vm = psutil.virtual_memory()
        avail_gb = vm.available / 1e9
        min_avail = min(min_avail, avail_gb)

        ws = commit = 0
        for p in tree():
            try:
                mi = p.memory_full_info()
                ws += getattr(mi, "wset", mi.rss)
                commit += getattr(mi, "pagefile", getattr(mi, "vms", 0))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        peak_ws = max(peak_ws, ws)
        peak_commit = max(peak_commit, commit)

        if avail_gb < threshold_gb:
            killed = True
            for p in tree():
                try:
                    p.kill()
                except psutil.Error:
                    pass
            try:
                proc.kill()
            except OSError:
                pass
            print(f"GUARD KILLED pid={proc.pid} at avail={avail_gb:.2f}GB "
                  f"(t+{time.time()-t0:.0f}s)", flush=True)
            break

        tick += 1
        if tick % 5 == 0:
            print(f"  [t+{time.time()-t0:4.0f}s] sys_avail={avail_gb:5.1f}GB  "
                  f"tree_ws={ws/1e9:.2f}GB tree_commit={commit/1e9:.2f}GB", flush=True)
        time.sleep(poll)

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass

    report = {
        "target_pid": proc.pid,
        "killed": killed,
        "min_sys_avail_gb": round(min_avail, 2),
        "peak_proc_ws_gb": round(peak_ws / 1e9, 3),
        "peak_proc_commit_gb": round(peak_commit / 1e9, 3),
    }
    print(f"GUARD report: {report}", flush=True)
    return report


def main():
    if "--" not in sys.argv:
        print("usage: mem_guard.py <threshold_gb> -- <command ...>")
        sys.exit(2)
    sep = sys.argv.index("--")
    threshold = float(sys.argv[1]) if sep > 1 else 8.0
    argv = sys.argv[sep + 1:]
    if not argv:
        print("no command after --")
        sys.exit(2)
    run_guarded(argv, threshold_gb=threshold)


if __name__ == "__main__":
    main()

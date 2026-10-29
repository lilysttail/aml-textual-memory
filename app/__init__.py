"""The service package.

Before anything imports numpy: its linear-algebra library starts one thread per CPU it can SEE, and in a
container that is the host's CPU count, not the container's quota (on a 2-CPU quota it saw 28). Those threads
burn the quota in a moment and the kernel then pauses the whole container — a similarity pass that takes 8 ms
on one thread took ~400 ms, and every other request waited with it. The service runs its heavy work on a few
threads of its own (SEARCH_HEAVY_CONCURRENCY), so the library gets one thread each. A value set in the
environment is kept.
"""
import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

"""mem_guard watches the launched child by PID, not by command-line match
(ke/栖 review: the old string match also matched the guard's own argv, so it
reported a phantom peak with no bench running).

With a tiny threshold the guard never kills, so these just check what it reports:
  - a child that holds ~120MB -> peak reflects the child
  - a trivial child that holds nothing -> peak stays ~0 (no phantom / not self)
"""
import sys

import pytest

from eval.mem_guard import run_guarded


def test_guard_captures_child_memory():
    code = "import time; x=bytearray(120*1024*1024); time.sleep(1.5); del x"
    rep = run_guarded([sys.executable, "-c", code],
                      threshold_gb=0.001, poll=0.2, max_s=30)
    assert rep["killed"] is False
    assert rep["peak_proc_ws_gb"] > 0.05, f"should capture child's ~120MB: {rep}"


def test_guard_no_phantom_peak_for_trivial_child():
    rep = run_guarded([sys.executable, "-c", "pass"],
                      threshold_gb=0.001, poll=0.2, max_s=30)
    # a child that holds nothing must not produce a large peak (and never the
    # guard's own footprint)
    assert rep["peak_proc_ws_gb"] < 0.05, f"trivial child -> tiny peak: {rep}"

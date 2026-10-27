"""旧尺子的单独入口（eval/locomo_recall）和代码赛道旧结果（eval/results）已退役：新尺子（eval/ruler）是唯一入口。"""
import importlib
import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_the_old_entry_and_the_old_results_are_gone():
    assert not (ROOT / "eval" / "locomo_recall").exists()
    assert not (ROOT / "eval" / "results").exists()
    assert not (ROOT / "tests" / "test_locomo_recall.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("eval.locomo_recall")


def test_nothing_in_the_repo_still_points_at_the_old_entry():
    me = pathlib.Path(__file__).resolve()
    stale = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".pytest_cache")]
        for name in filenames:
            p = pathlib.Path(dirpath) / name
            if p == me or p.suffix not in (".py", ".md", ".txt", ".yml", ".yaml", ".toml", ".cfg", ""):
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            if "eval/locomo_recall" in text.replace("\\", "/") or "eval.locomo_recall" in text:
                stale.append(str(p.relative_to(ROOT)))
    assert stale == []


def test_the_ruler_readme_says_what_the_core_engine_is_for_and_which_number_is_the_baseline():
    text = (ROOT / "eval" / "ruler" / "README.md").read_text(encoding="utf-8")
    assert "CoreTarget" in text and "reproduc" in text.lower()
    assert "0.7899" in text and "baseline" in text.lower()
    assert "turn_index" in text

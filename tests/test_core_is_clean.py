"""核心目录里搜不到比赛专用的东西。

禁的只有两类：写死的模型名 / 供应商名，和比赛专用配置 / 赛道名。
通用参数名（api_key、base_url、model、dim）不禁——核心的通用 HTTP 嵌入
适配器要接 api_key，禁了只能绕词。
"""
import os
import re

CORE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")

# 写死的模型名 / 供应商名，比赛专用配置名（大小写如它们在 env 里那样）。
BANNED = [
    "text-embedding-v4",
    "gpt-4o",
    "dashscope",
    "TWO_LEVEL",
    "TASK_PATHS",
    "TASK_KEY",
    "RECALL_PATHS",
    "RERANK_N",
    "FINGERPRINT_STEM_IDENT",
]

# 赛道名 / 评测平台：按整词匹配（"answered" 里没有 swe，"encoding" 里没有 coding）。
BANNED_WORDS = [
    "aml",
    "agent-memory-challenge",
    "coding",
    "textual",
    "leaderboard",
    "swe",
    "bench",
]

ALLOWED_NOT_BANNED = ["api_key", "base_url", "model", "dim"]


def _core_files():
    for root, dirs, files in os.walk(CORE_DIR):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in sorted(files):
            if name.endswith((".py", ".md", ".txt", ".json")):
                yield os.path.join(root, name)


def test_the_core_carries_no_contest_specific_word():
    hits = []
    for path in _core_files():
        rel = os.path.relpath(path, CORE_DIR)
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                for word in BANNED:
                    if word in line:
                        hits.append(f"{rel}:{i}: {word} :: {line.strip()[:90]}")
                low = line.lower()
                for word in BANNED_WORDS:
                    if re.search(rf"\b{re.escape(word)}\b", low):
                        hits.append(f"{rel}:{i}: {word} :: {line.strip()[:90]}")
    assert not hits, ("the core must not mention the contest's own things:\n"
                      + "\n".join(hits))


def test_the_core_does_not_import_the_shell():
    """核心不 import 任何 app.*：它要能整份搬走（L6）。"""
    import re
    pat = re.compile(r"^\s*(from|import)\s+app\b", re.MULTILINE)
    hits = []
    for path in _core_files():
        if not path.endswith(".py"):
            continue
        with open(path, encoding="utf-8") as f:
            if pat.search(f.read()):
                hits.append(os.path.relpath(path, CORE_DIR))
    assert not hits, f"the core must not import the shell: {hits}"


def test_the_core_reads_no_environment_variable():
    import re
    pat = re.compile(r"os\.environ|getenv\(")
    hits = []
    for path in _core_files():
        if not path.endswith(".py"):
            continue
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if pat.search(line):
                    hits.append(f"{os.path.relpath(path, CORE_DIR)}:{i}")
    assert not hits, f"the core must not read the environment: {hits}"

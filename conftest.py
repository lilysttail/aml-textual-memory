"""仓库根的 conftest：拿错 Python 跑测试，当场失败。

这个仓部署在 Dockerfile 写着的那个 Python 上（`FROM python:3.11-slim`）。
3.11 在**定义函数时**就求值类型注解，3.12+ 不求值（PEP 649）——所以同一份
代码在 3.14 上是绿的，在 3.11 上可能 import 就炸（少一个 typing import 就
让整个包加载不了）。这种事不能靠记性：这里直接读 Dockerfile 核对版本，
对不上就报错退出。

跑测试请用项目自己的解释器（比赛仓的 .venv_scaffold）。
"""
import pathlib
import re
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
_DOCKERFILE = _ROOT / "Dockerfile"


def _deployed_version():
    """The Python version the service is deployed on (from the Dockerfile)."""
    if not _DOCKERFILE.exists():
        return None
    m = re.search(r"FROM\s+python:(\d+)\.(\d+)",
                  _DOCKERFILE.read_text(encoding="utf-8"))
    return (int(m.group(1)), int(m.group(2))) if m else None


_EXPECTED = _deployed_version()
if _EXPECTED is not None and sys.version_info[:2] != _EXPECTED:
    raise RuntimeError(
        "wrong Python for this repo: it is deployed on Python {}.{} (see Dockerfile), "
        "but the tests are running on {}.{} ({}). Python 3.11, 3.12 and 3.13 evaluate "
        "type annotations at definition time; only 3.14 stopped doing that, so a "
        "green run on 3.14 does not mean the service starts. Run the tests with the "
        "project's own interpreter (the .venv_scaffold virtualenv next to the "
        "scaffold).".format(
            _EXPECTED[0], _EXPECTED[1],
            sys.version_info.major, sys.version_info.minor, sys.executable))

"""core — 一份与数据来源无关的检索核心。

核心吃「记录」（Record），存的是带说话人、带时间、带会话 / 轮次的结构化行；
它提供写入、检索、补前后文、按时间排、快照换代五件事，不生成答案，也
不认识数据是从比赛服务来的，还是从别的数据源来的。

用法：

    from core import build, Record
    core = build(CoreConfig(db_path="..."))
    core.write([Record(record_id="r1", session_id="s1", speaker="alice",
                        ts="2026-10-21T05:00:00.000Z", text="...")],
               space_id="home", batch_id="b1")
    hits = core.search("那句话", space_id="home", top_k=5)

核心不 import 任何 app.*，不读环境变量，不含比赛专用的配置、模型名或
赛道名（L6：比赛交完之后，这个目录整份搬回家就能用）。
"""
from .config import CoreConfig
from .engine import Core
from .models import Record, SearchHit

__all__ = ["CoreConfig", "Core", "Record", "SearchHit", "build"]


def build(config=None, **kwargs):
    """构造核心。config 可以是 CoreConfig，也可以直接给关键字参数。"""
    if config is None:
        config = CoreConfig(**kwargs)
    elif kwargs:
        raise TypeError("give either a CoreConfig or keyword arguments, not both")
    return Core.build(config)

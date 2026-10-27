"""核心的记录模型：对外的唯一数据结构。

Record = 一条消息（另一个使用方：一行原话；比赛：/add 的一条 messages 元素）。
正文一字不改：核心存裸正文，契约渲染（拼 [时间] role: 前缀）是壳的事。
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Record:
    """一条记录。

    record_id  在一个 space 内唯一；主键是 (space_id, record_id) 复合键，
                不同的 space 可以用同一个 record_id（互不抢）。
    session_id 一段会话 / 一扇窗（另一个使用方 = 窗口号）。
    turn_index 会话内的位置；不是永久编号（删除 / 后补会让它重排），
                核心按「当前有效窗口顺序」取邻居。
    speaker    谁说的；核心只当不透明字符串，不解释。
    ts         事件时间（ISO-8601 UTC 毫秒）；created_at 是入库时间。
    valid      软删：False 的行搜不到，也不当邻居露出。
    """

    record_id: str
    session_id: str
    text: str
    turn_index: Optional[int] = None
    speaker: Optional[str] = None
    direction: Optional[str] = None
    line: Optional[str] = None
    ts: Optional[str] = None
    created_at: Optional[str] = None
    sec_flag: Optional[bool] = None
    valid: bool = True
    state: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    space_id: Optional[str] = None
    batch_id: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "session_id": self.session_id,
            "text": self.text,
            "turn_index": self.turn_index,
            "speaker": self.speaker,
            "direction": self.direction,
            "line": self.line,
            "ts": self.ts,
            "created_at": self.created_at if self.created_at is not None else self.ts,
            "sec_flag": None if self.sec_flag is None else int(bool(self.sec_flag)),
            "valid": int(bool(self.valid)),
            "state": self.state,
            "meta": None if self.meta is None else json.dumps(self.meta, ensure_ascii=False),
            "batch_id": self.batch_id,
        }

    @classmethod
    def from_row(cls, row) -> "Record":
        """从一行数据库记录还原（列名与 to_row 对应）。"""
        data = dict(row)
        meta = data.get("meta")
        if meta:
            try:
                data["meta"] = json.loads(meta)
            except (ValueError, TypeError):
                data["meta"] = None
        sec = data.get("sec_flag")
        data["sec_flag"] = None if sec is None else bool(sec)
        data["valid"] = bool(data.get("valid", 1))
        return cls(**data)


@dataclass
class SearchHit:
    """一条命中 + 它的邻居。

    neighbors_before / neighbors_after 是当前有效的紧邻记录（不带分数）；
    怎么把它们并进结果，是壳的渲染层的事。
    """

    record_id: str
    space_id: str
    session_id: str
    text: str
    score: float
    turn_index: Optional[int] = None
    speaker: Optional[str] = None
    direction: Optional[str] = None
    line: Optional[str] = None
    ts: Optional[str] = None
    created_at: Optional[str] = None
    path_scores: Dict[str, float] = field(default_factory=dict)
    neighbors_before: List[Record] = field(default_factory=list)
    neighbors_after: List[Record] = field(default_factory=list)

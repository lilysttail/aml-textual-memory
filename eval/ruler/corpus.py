"""Reading the public LoCoMo-Refined copy into plain objects, and cutting a conversation into write blocks.

The cut is the one the first ruler used (retired 2026-10-23, see README.md): at most 20 messages or 2,000 words per block, one block never
crosses a session, and the ids are the same — so the new ruler can reproduce the old numbers. The pinned numbers below are the one
place the copy is pinned; eval/local_load/test_full_corpus.py imports them from here.
"""
import datetime as dt
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Iterator, List, Tuple

PINNED_QUESTIONS = 1382
PINNED_QUESTIONS_SHA256 = "4dd84cf65a28ece0ed6b2a1b7b2700ea639f4e7b2c5e1afcc511e47e81732790"
PINNED_CONVERSATIONS_SHA256 = "abbe220013815b1761c07d9ab7d6147ee08fb089f1a8a083fd31188cdcd91444"

_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
     "December"], start=1)}
_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2}) (am|pm) on (\d{1,2}) ([A-Za-z]+), (\d{4})$")


class PinMismatch(Exception):
    """The files are not the pinned copy (question count or a sha256 differs)."""


def session_time_ms(text: str) -> int:
    """'1:56 pm on 8 May, 2023' -> epoch milliseconds, read as UTC (the data carries no time zone)."""
    m = _TIME_RE.match(text or "")
    if not m:
        raise ValueError("unreadable session time")
    hour, minute, ampm, day, month, year = m.groups()
    hour, minute, day, year = int(hour), int(minute), int(day), int(year)
    if month not in _MONTHS or not 1 <= hour <= 12 or minute > 59:
        raise ValueError("unreadable session time")
    hour = hour % 12 + (12 if ampm == "pm" else 0)
    return int(dt.datetime(year, _MONTHS[month], day, hour, minute, tzinfo=dt.timezone.utc).timestamp() * 1000)


@dataclass(frozen=True)
class Msg:
    dia_id: str
    speaker: str
    role: str
    text: str
    caption: str
    session_index: int
    message_index: int


@dataclass
class Session:
    index: int
    time_text: str
    time_ms: int
    messages: List[Msg] = field(default_factory=list)


@dataclass
class Conv:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: List[Session] = field(default_factory=list)


@dataclass
class Corpus:
    conversations: List[Conv]
    questions: List[dict]
    pinned: bool


@dataclass
class Block:
    """One Add call's worth of messages, all from one session."""
    user_id: str
    session_id: str
    request_id: str
    session: Session
    messages: List[Msg]

    def pairs(self) -> Iterator[Tuple[Msg, Session]]:
        for m in self.messages:
            yield m, self.session


def user_id_of(sample_id: str) -> str:
    return f"eval:local:locomo_refined:{sample_id}"


def data_paths(root: str) -> dict:
    pub = os.path.join(root, "data", "public")
    return {"questions": os.path.join(pub, "questions.jsonl"),
            "conversations": os.path.join(pub, "conversations.jsonl")}


def _read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_corpus(root: str, require_pin: bool = True) -> Corpus:
    """Read <root>/data/public/{conversations,questions}.jsonl. `require_pin` refuses anything but the pinned copy."""
    paths = data_paths(root)
    raw_convs = _read_jsonl(paths["conversations"])
    questions = _read_jsonl(paths["questions"])
    pinned = (len(questions) == PINNED_QUESTIONS
              and _sha256(paths["questions"]) == PINNED_QUESTIONS_SHA256
              and _sha256(paths["conversations"]) == PINNED_CONVERSATIONS_SHA256)
    if require_pin and not pinned:
        raise PinMismatch("the data is not the pinned copy (question count / sha256 differ)")
    convs = []
    for c in raw_convs:
        sessions = []
        for s in c.get("sessions") or []:
            msgs = [Msg(dia_id=str(m.get("dia_id")), speaker=m.get("speaker") or "unknown", role=m.get("role") or "user",
                        text=m.get("text") or "", caption=(m.get("blip_caption") or "").strip(),
                        session_index=int(s.get("session_index", 0)), message_index=int(m.get("message_index", i)))
                    for i, m in enumerate(s.get("messages") or [])]
            sessions.append(Session(index=int(s.get("session_index", 0)), time_text=s.get("date_time") or "",
                                    time_ms=session_time_ms(s.get("date_time") or ""), messages=msgs))
        convs.append(Conv(sample_id=c.get("sample_id") or "sample", speaker_a=c.get("speaker_a") or "",
                          speaker_b=c.get("speaker_b") or "", sessions=sessions))
    return Corpus(conversations=convs, questions=questions, pinned=pinned)


def count_words(text: str) -> int:
    return len([w for w in (text or "").split() if w])


def chunk_conversation(conv: Conv, max_messages: int = 20, max_words: int = 2000) -> List[Block]:
    """The old ruler's cut: <= max_messages messages and <= max_words words per block; a block never crosses a session."""
    user_id = user_id_of(conv.sample_id)
    blocks: List[Block] = []
    for session in conv.sessions:
        session_id = f"{conv.sample_id}:{session.index}"
        chunk: List[Msg] = []
        words = 0
        chunk_index = 0

        def flush():
            nonlocal chunk, words, chunk_index
            if not chunk:
                return
            blocks.append(Block(user_id=user_id, session_id=session_id,
                                request_id=f"{user_id}:{session.index}:{chunk_index}", session=session, messages=chunk))
            chunk, words = [], 0
            chunk_index += 1

        for m in session.messages:
            w = count_words(m.text)
            if chunk and (len(chunk) >= max_messages or words + w > max_words):
                flush()
            chunk.append(m)
            words += w
        flush()
    return blocks

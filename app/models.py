"""Pydantic models for the AML Add/Search contract.

Source: participant implementation of the official AML contract
(https://github.com/tlysanhuo/agent-memory-challenge) - the first-cycle
textual-track entry, MIT licence. That repository was no longer publicly
reachable when we checked on 2026-10-27; THIRD_PARTY_NOTICES.md says what we
took from it. Checked item by item against the official API Guide
(https://agentmemoryleaderboard.ai/api-guide) on 2026-10-27.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

# The official contract says `timestamp` is optional Unix milliseconds. We
# refuse anything outside the range a date can carry (a bigger number would
# blow up the renderer instead of being rejected), and we refuse empty
# messages and empty content — the contract says content is non-empty.
_MAX_TIMESTAMP_MS = 4102444800000          # 2100-01-01T00:00:00Z


class Message(BaseModel):
    role: str                      # e.g. "user" / "assistant"
    timestamp: Optional[int] = Field(default=None, ge=0, le=_MAX_TIMESTAMP_MS)
                                     # epoch milliseconds; the official contract
                                     # says it is optional — a message may arrive
                                     # without a time
    content: str = Field(min_length=1)   # memory text

    @field_validator("content")
    @classmethod
    def _content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be blank")
        return v


class AddRequest(BaseModel):
    request_id: str = Field(min_length=1)
    messages: List[Message] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class AddResponse(BaseModel):
    success: bool = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    options: Optional[List[str]] = None   # only present for multiple-choice items
    user_id: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1)  # the response must not exceed it


class SearchHit(BaseModel):
    id: str
    content: str                   # the candidate as stored: "[YYYY-MM-DD HH:MM]
                                   # role: text" when the event time is known,
                                   # "[role] text" when it is not (no made-up
                                   # date). The answering model reads this
                                   # verbatim, so it is the candidate itself —
                                   # the neighbours are separate candidates
                                   # (behind NEIGHBOUR_CANDIDATES, on by
                                   # default).
    score: float
    created_at: str                # ISO-8601


class SearchResponse(BaseModel):
    data: List[SearchHit]

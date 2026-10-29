"""Pydantic models for the AML Add/Search contract.

Source: participant implementation of the official AML contract
(https://github.com/tlysanhuo/agent-memory-challenge) - the first-cycle
textual-track entry, MIT licence. That repository was no longer publicly
reachable when we checked on 2026-10-27; THIRD_PARTY_NOTICES.md says what we
took from it. Checked item by item against the official API Guide
(https://agentmemoryleaderboard.ai/api-guide) on 2026-10-27.
"""
from typing import Any, List, Optional
from pydantic import BaseModel

# 2026-10-29: the formal evaluation stopped on our own 422 — this model was STRICTER than the official contract (it refused
# blank content, a timestamp before 1970 or after 2100 or with a fraction, an empty session id, an empty message list ...), and the
# platform does not retry a 422. The contract is now the only bar: a field the contract requires must be PRESENT (a missing
# request_id / user_id / session_id / messages is still a 422, logged by field name only); everything else is taken as it comes and
# made usable in the shell (app/main.py, `normalize_message`): a message whose content carries no text is skipped, not the batch;
# a role that is missing becomes "user"; a timestamp that cannot be read as a date is "no time". Retrieval is unchanged.
class AddRequest(BaseModel):
    request_id: str                # required, may be empty (returned as given; the batch is then keyed by its content)
    messages: List[Any]            # required, may be empty (nothing written). Each message: role (missing / not a string ->
                                   # "user"), timestamp (epoch ms, optional; see app/main.py), content (blank / null -> the message
                                   # is skipped; a list of text parts is joined with spaces); an item that is not an object is skipped
    user_id: str                   # required, may be empty (returned as given)
    session_id: str                # required, may be empty (the batch's request_id is then its session key)


class AddResponse(BaseModel):
    success: bool = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: str                     # empty / blank -> 200 with no results (no stricter than the contract)
    options: Optional[List[str]] = None   # only present for multiple-choice items
    user_id: str                   # an ordinary space name, "" included: searches what was added under exactly this value
    top_k: int = 10                # below 1 -> 200 with no results; the response never exceeds it


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

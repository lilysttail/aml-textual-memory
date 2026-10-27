"""The three facts about the public copy that decide how the ruler must store it (step 1). Numbers only."""
import re
from collections import Counter

_PATTERN = re.compile(r"^\d{1,2}:\d{2} (am|pm) on \d{1,2} [A-Za-z]+, \d{4}$")


def _word(text: str, name: str) -> bool:
    return re.search(r"(?<![A-Za-z])" + re.escape(name) + r"(?![A-Za-z])", text or "") is not None


def compute(corpus, raw_conversations) -> dict:
    """`raw_conversations` = the conversation records as read from the file (the time-level facts need the raw keys)."""
    msgs = [(c, s, m) for c in raw_conversations for s in c["sessions"] for m in s["messages"]]
    out = {"messages_total": len(msgs), "sessions_total": sum(len(c["sessions"]) for c in raw_conversations)}
    # 1. which level carries the time
    out["time_on_message_own_key_date_time"] = sum(1 for _c, _s, m in msgs if "date_time" in m)
    out["messages_with_session_date_time_copy"] = sum(1 for _c, _s, m in msgs if m.get("session_date_time"))
    out["messages_whose_copy_equals_their_session_time"] = sum(1 for _c, s, m in msgs if m.get("session_date_time") == s.get("date_time"))
    out["sessions_with_date_time"] = sum(1 for c in raw_conversations for s in c["sessions"] if s.get("date_time"))
    out["sessions_where_all_messages_share_one_time"] = sum(
        1 for c in raw_conversations for s in c["sessions"] if len({m.get("session_date_time") for m in s["messages"]}) == 1)
    out["session_times_matching_the_pattern"] = sum(1 for c in raw_conversations for s in c["sessions"] if _PATTERN.match(s.get("date_time") or ""))
    out["conversations_with_duplicate_session_times"] = sum(
        1 for c in raw_conversations if len({s["date_time"] for s in c["sessions"]}) < len(c["sessions"]))
    out["roles"] = dict(Counter(m["role"] for _c, _s, m in msgs))
    # 2. picture descriptions
    out["messages_with_nonempty_caption"] = sum(1 for _c, _s, m in msgs if (m.get("blip_caption") or "").strip())
    out["messages_with_nonempty_images_list"] = sum(1 for _c, _s, m in msgs if m.get("images"))
    out["questions_marked_multimodal"] = sum(1 for q in corpus.questions if q.get("is_multi_modality"))
    # 3. questions that name a person
    speakers = {c.speaker_a for c in corpus.conversations} | {c.speaker_b for c in corpus.conversations}
    own = other = none = 0
    for q in corpus.questions:
        a, b, t = q.get("speaker_a") or "", q.get("speaker_b") or "", q.get("question") or ""
        if (a and _word(t, a)) or (b and _word(t, b)):
            own += 1
        elif any(_word(t, n) for n in speakers if n not in (a, b)):
            other += 1
        else:
            none += 1
    out["questions_total"] = len(corpus.questions)
    out["questions_naming_a_speaker_of_their_own_conversation"] = own
    out["questions_naming_only_a_speaker_of_another_conversation"] = other
    out["questions_naming_no_known_speaker"] = none
    return out

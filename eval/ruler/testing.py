"""A tiny synthetic corpus in the LoCoMo-Refined file layout, for the ruler's own tests.

Nothing here is the real data: two made-up conversations, a few questions of each category, one with a name that
never appears in any message text, one asking for a time, one whose answer lives only in an image caption.
`write_mini_corpus(dir)` writes <dir>/data/public/{questions,conversations}.jsonl like a commit copy of the real set.
"""
import json
import os

# conversation 1: Alice (user) and Bob (assistant); conversation 2: Cara and Dan
_CONVS = [
    {
        "sample_id": "mini-1", "conversation_idx": 0, "speaker_a": "Alice", "speaker_b": "Bob",
        "sessions": [
            {"session_index": 1, "date_time": "1:56 pm on 8 May, 2023", "messages": [
                {"dia_id": "D1:1", "speaker": "Alice", "role": "user", "text": "I adopted a puppy named Biscuit last week.", "blip_caption": ""},
                {"dia_id": "D1:2", "speaker": "Bob", "role": "assistant", "text": "That is wonderful news, congratulations.", "blip_caption": ""},
                {"dia_id": "D1:3", "speaker": "Alice", "role": "user", "text": "Look at this picture of the garden.", "blip_caption": "a photo of a red tulip garden"},
            ]},
            {"session_index": 2, "date_time": "12:05 am on 1 January, 2024", "messages": [
                {"dia_id": "D2:1", "speaker": "Bob", "role": "assistant", "text": "How is the new job going?", "blip_caption": ""},
                {"dia_id": "D2:2", "speaker": "Alice", "role": "user", "text": "It is going well, I started the pottery class too.", "blip_caption": ""},
            ]},
        ],
    },
    {
        "sample_id": "mini-2", "conversation_idx": 1, "speaker_a": "Cara", "speaker_b": "Dan",
        "sessions": [
            {"session_index": 1, "date_time": "12:30 pm on 15 March, 2022", "messages": [
                {"dia_id": "D1:1", "speaker": "Cara", "role": "user", "text": "We visited the lighthouse on the coast.", "blip_caption": ""},
                {"dia_id": "D1:2", "speaker": "Dan", "role": "assistant", "text": "Nice, I love lighthouses.", "blip_caption": "a white lighthouse at sunset"},
            ]},
        ],
    },
]

_QUESTIONS = [
    # q1: names Alice, evidence D1:1; the name Alice never appears inside any message text of conversation 1
    {"qa_id": "q1", "sample_id": "mini-1", "conversation_idx": 0, "qa_index": 0, "speaker_a": "Alice", "speaker_b": "Bob",
     "question": "What did Alice adopt?", "answer": ["a puppy", "Biscuit"], "category": "4", "is_multi_modality": False, "evidence": ["D1:1"]},
    # q2: a question about time (category 2), evidence D2:2
    {"qa_id": "q2", "sample_id": "mini-1", "conversation_idx": 0, "qa_index": 1, "speaker_a": "Alice", "speaker_b": "Bob",
     "question": "When did Alice start the pottery class?", "answer": ["1 January 2024"], "category": "2", "is_multi_modality": False, "evidence": ["D2:2"]},
    # q3: the answer is only in a caption (evidence D1:3 carries it), no name in the question
    {"qa_id": "q3", "sample_id": "mini-1", "conversation_idx": 0, "qa_index": 2, "speaker_a": "Alice", "speaker_b": "Bob",
     "question": "What flowers were in the picture?", "answer": ["red tulip", "tulips"], "category": "1", "is_multi_modality": True, "evidence": ["D1:3"]},
    # q4: another conversation, names Dan, evidence D1:2
    {"qa_id": "q4", "sample_id": "mini-2", "conversation_idx": 1, "qa_index": 0, "speaker_a": "Cara", "speaker_b": "Dan",
     "question": "Which building did Cara visit?", "answer": ["the lighthouse"], "category": "4", "is_multi_modality": False, "evidence": ["D1:1"]},
    # q5: not a question about a person, no time wording (category 3)
    {"qa_id": "q5", "sample_id": "mini-2", "conversation_idx": 1, "qa_index": 1, "speaker_a": "Cara", "speaker_b": "Dan",
     "question": "Are lighthouses useful for ships?", "answer": ["yes"], "category": "3", "is_multi_modality": False, "evidence": ["D1:2"]},
]


def mini_conversations():
    """The conversations in the real file layout (every key the real files carry that the ruler reads)."""
    out = []
    for c in _CONVS:
        conv = {k: v for k, v in c.items() if k != "sessions"}
        sessions = []
        for s in c["sessions"]:
            msgs = []
            for i, m in enumerate(s["messages"]):
                msgs.append({"session_index": s["session_index"], "session_date_time": s["date_time"], "message_index": i,
                             "dia_id": m["dia_id"], "speaker": m["speaker"], "role": m["role"], "text": m["text"],
                             "images": [], "blip_caption": m["blip_caption"], "query": "",
                             "has_multimodal_context": bool(m["blip_caption"])})
            sessions.append({"session_index": s["session_index"], "date_time": s["date_time"], "messages": msgs})
        conv["sessions"] = sessions
        conv["session_count"] = len(sessions)
        conv["message_count"] = sum(len(s["messages"]) for s in sessions)
        out.append(conv)
    return out


def mini_questions():
    return [dict(q, evidence_messages=[]) for q in _QUESTIONS]


def write_mini_corpus(root):
    """Write the mini corpus under <root>/data/public/ and return <root> (the "commit copy" root)."""
    pub = os.path.join(root, "data", "public")
    os.makedirs(pub, exist_ok=True)
    with open(os.path.join(pub, "conversations.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for c in mini_conversations():
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(os.path.join(pub, "questions.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for q in mini_questions():
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return root


# ---- a fake embedding provider (the DashScope HTTP shape), for tests that must not touch the network ----

import hashlib
import json

FAKE_KEY = "test-key-NOT-A-REAL-KEY-123"
DIM = 1024
SYNONYMS = {"wedding": "married", "ceremony": "married"}          # 假向量里“结婚”和“婚礼”落在同一格


def vec_for(text):
    v = [0.0] * DIM
    for w in "".join(ch if ch.isalnum() else " " for ch in text.lower()).split():
        w = SYNONYMS.get(w, w)
        v[int(hashlib.sha256(w.encode()).hexdigest(), 16) % DIM] += 1.0
    if not any(v):
        v[0] = 1.0
    return v


class FakeEmbedApi:
    """顶替 DashScope 的 HTTP：按 OpenAI 兼容的形状回，每个字 1 个 token。"""

    def __init__(self):
        self.calls = 0
        self.texts = 0

    def __call__(self, url, headers, body, timeout):
        assert headers["Authorization"] == "Bearer " + FAKE_KEY
        inputs = json.loads(body)["input"]
        self.calls += 1
        self.texts += len(inputs)
        data = [{"index": i, "embedding": vec_for(t)} for i, t in enumerate(inputs)]
        return 200, json.dumps({"usage": {"total_tokens": sum(len(t.split()) for t in inputs)}, "data": data}).encode("utf-8")

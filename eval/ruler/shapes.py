"""What one Add message looks like, in the four shapes the platform might send. Defined here and nowhere else.

The platform's Add carries, per message: `role` (user / assistant), `content` (plain text) and an optional `timestamp`
(epoch milliseconds). It is not public whether it also carries the speaker's name or a picture description, so the ruler
measures every case:

    a  the most conservative: role + plain text. No time, no name, no picture description.
    b  a + `timestamp` (milliseconds). The data has a time ONLY on the session, so every message of a session gets the same
       millisecond (the session's time, read as UTC); the order inside a session is the order of the messages.
    c  b + the speaker's name in front of the text:  "Alice: <text>"
    d  c + the picture description, where the message has one:  "Alice: <text> [image: <description>]"

`legacy` is what the old ruler wrote (role = the speaker's name, plain text, no time). It exists only so the old numbers
can be reproduced; it is not one of the four.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    doc: str
    time: bool
    names: bool
    captions: bool
    role_is_speaker: bool = False


SHAPES = {
    "a": ShapeSpec("a", "role + plain text", time=False, names=False, captions=False),
    "b": ShapeSpec("b", "a + millisecond timestamp (the session's time)", time=True, names=False, captions=False),
    "c": ShapeSpec("c", "b + speaker name before the text", time=True, names=True, captions=False),
    "d": ShapeSpec("d", "c + picture description", time=True, names=True, captions=True),
}
LEGACY = ShapeSpec("legacy", "what the old ruler wrote (role = speaker name, no time)", time=False, names=False,
                   captions=False, role_is_speaker=True)
# the old ruler also stored the session time (as text, in `ts`); the service only takes milliseconds, so "legacy_t" is the same
# thing with the session time as a millisecond timestamp. Used to find out whether the time alone moves the old numbers.
LEGACY_T = ShapeSpec("legacy_t", "legacy + the session time as a millisecond timestamp", time=True, names=False,
                     captions=False, role_is_speaker=True)
ALL_SHAPES = dict(SHAPES, legacy=LEGACY, legacy_t=LEGACY_T)

IMAGE_NOTE = "{content} [image: {caption}]"


def render(shape: str, msg, session) -> dict:
    """One message -> the dict the platform's Add would carry for it. KeyError for an unknown shape."""
    spec = ALL_SHAPES[shape]
    content = msg.text
    if spec.names:
        content = f"{msg.speaker}: {content}"
    if spec.captions and msg.caption:
        content = IMAGE_NOTE.format(content=content, caption=msg.caption)
    out = {"role": msg.speaker if spec.role_is_speaker else msg.role, "content": content}
    if spec.time:
        out["timestamp"] = session.time_ms
    return out

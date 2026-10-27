"""Claude Haiku as a cross-check answerer, called as a program: one `claude -p` process per question (step 2).

One question per call, on purpose: the questions of one conversation share one memory store, and putting several into one call
lets the model answer one question with another question's memories. The question and the memories go in on standard input
(never on the command line, never through a person). Everything that could add something of ours is switched off:

    --tools ""                  no tool at all                 --setting-sources ""   no user / project / local settings (no home hooks)
    --strict-mcp-config         no MCP server                  --disable-slash-commands   no skills
    --no-session-persistence    no conversation record         --system-prompt <one line>   Claude Code's own system prompt replaced
    working directory           an empty directory (no CLAUDE.md can be found)
The first line of the prompt says it is a test and forbids tools. After each call the JSON is checked: the model that answered
must be the one asked for, and the number of turns beyond the first plus denied tool requests must be 0 — otherwise the row is
void. Runs on the subscription, so no API money; the JSON's own token counts and list-price figure are kept for the report.
"""
import json
import os
import shutil
import subprocess
from typing import Callable, Optional, Sequence, Tuple

from . import answerer
from .chat import ChatError, ChatResult

EXECUTABLE = "claude"
SYSTEM_PROMPT = "You answer a question using only the text you are given."


CHILD_ENV = {"DISABLE_AUTOUPDATER": "1"}          # Claude Code's own switch: the program must not update itself in the middle of a run (step 4 claims its version once)


def run_process(cmd: Sequence[str], stdin: str, cwd: str, timeout: float) -> Tuple[int, str, str]:
    env = dict(os.environ, **CHILD_ENV)                # the child's environment only: this process's own is not touched
    p = subprocess.run(list(cmd), input=stdin, capture_output=True, text=True, encoding="utf-8", cwd=cwd, timeout=timeout, env=env)
    return p.returncode, p.stdout, p.stderr


class ClaudeCliAnswerer:
    def __init__(self, model: str, run: Callable = run_process, workdir: Optional[str] = None, timeout_s: float = 180.0,
                 executable: Optional[str] = None):
        self.model = model
        self._run = run
        self.workdir = workdir
        self.timeout_s = timeout_s
        self.executable = executable or shutil.which(EXECUTABLE) or EXECUTABLE

    def command(self) -> list:
        return [self.executable, "-p", "--model", self.model, "--output-format", "json",
                "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence",
                "--disable-slash-commands", "--system-prompt", SYSTEM_PROMPT]

    def __call__(self, question: str, memories: Sequence[str]) -> ChatResult:
        if self.workdir is None or os.listdir(self.workdir):
            raise ValueError("the work directory must be an existing, empty directory")
        stdin = answerer.build_answer_prompt(memories, question, prefix=answerer.HAIKU_TEST_LINE)
        cmd = self.command()
        rc, out, _err = self._run(cmd, stdin, self.workdir, self.timeout_s)
        if rc != 0:
            raise ChatError(f"claude exited with {rc}")
        try:
            doc = json.loads(out)
            if doc.get("is_error"):
                raise ChatError("claude reported an error", status=doc.get("api_error_status"))
            text = doc["result"]
            usage = doc["usage"]
            models = list(doc["modelUsage"])
            turns = int(doc["num_turns"])
            denied = len(doc.get("permission_denials") or [])
        except ChatError:
            raise
        except (ValueError, KeyError, TypeError):
            raise ChatError("the claude output could not be read")
        if models != [self.model]:
            raise ChatError("another model answered than the one asked for")
        i, cc, cr = usage.get("input_tokens", 0), usage.get("cache_creation_input_tokens", 0), usage.get("cache_read_input_tokens", 0)
        return ChatResult(text=str(text), prompt_tokens=i + cc + cr, completion_tokens=usage.get("output_tokens", 0), model=models[0],
                          extra={"tool_calls": max(0, turns - 1) + denied, "num_turns": turns, "cost_usd": doc.get("total_cost_usd"),
                                 "input_tokens": i, "cache_creation_tokens": cc, "cache_read_tokens": cr})

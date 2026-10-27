"""Haiku 交叉核那一路：程序里一题一次调 `claude -p`（假的进程，不起真的 claude）。"""
import json
import os

import pytest

from eval.ruler import answerer, chat, claude_cli

MODEL = "claude-haiku-5-5"


def good(result="Paris", turns=1, denials=None, usage=None, models=None, is_error=False):
    return json.dumps({
        "type": "result", "subtype": "success", "is_error": is_error, "result": result, "num_turns": turns,
        "permission_denials": denials or [], "total_cost_usd": 0.0002, "api_error_status": None,
        "usage": usage or {"input_tokens": 2, "cache_creation_input_tokens": 900, "cache_read_input_tokens": 100, "output_tokens": 3},
        "modelUsage": models or {MODEL: {"inputTokens": 2, "outputTokens": 3}}})


class FakeRun:
    def __init__(self, *outs):
        self.outs = list(outs)
        self.calls = []

    def __call__(self, cmd, stdin, cwd, timeout):
        self.calls.append({"cmd": cmd, "stdin": stdin, "cwd": cwd, "cwd_listing": sorted(os.listdir(cwd)), "timeout": timeout})
        rc, out, err = self.outs.pop(0)
        return rc, out, err


def make(tmp_path, *outs):
    run = FakeRun(*outs)
    return claude_cli.ClaudeCliAnswerer(MODEL, run=run, workdir=str(tmp_path), executable=claude_cli.EXECUTABLE), run


def test_the_command_has_every_lockdown_switch_and_the_prompt_goes_in_on_standard_input(tmp_path):
    a, run = make(tmp_path, (0, good(), ""))
    r = a("Where?", ["[x] a", "[y] b"])
    c = run.calls[0]
    cmd = c["cmd"]
    assert cmd[0] == claude_cli.EXECUTABLE and "-p" in cmd
    assert cmd[cmd.index("--model") + 1] == MODEL
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--tools") + 1] == ""                       # 一个工具都不给
    assert "--strict-mcp-config" in cmd and "--no-session-persistence" in cmd and "--disable-slash-commands" in cmd
    assert cmd[cmd.index("--setting-sources") + 1] == ""             # 不读用户、项目、本地设置（本机的钩子进不来）
    assert cmd[cmd.index("--system-prompt") + 1] == claude_cli.SYSTEM_PROMPT
    assert c["stdin"] == answerer.build_answer_prompt(["[x] a", "[y] b"], "Where?", prefix=answerer.HAIKU_TEST_LINE)
    assert "Where?" not in " ".join(cmd) and "[x] a" not in " ".join(cmd)       # 题和记忆不上命令行
    assert (r.text, r.completion_tokens, r.model) == ("Paris", 3, MODEL)


def test_it_runs_in_an_empty_directory(tmp_path):
    a, run = make(tmp_path, (0, good(), ""))
    a("Q?", [])
    assert run.calls[0]["cwd_listing"] == []


def test_a_nonempty_work_directory_is_refused(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("x", encoding="utf-8")
    a, run = make(tmp_path, (0, good(), ""))
    with pytest.raises(ValueError):
        a("Q?", [])
    assert run.calls == []


def test_the_prompt_tokens_count_everything_the_model_read(tmp_path):
    a, _ = make(tmp_path, (0, good(), ""))
    r = a("Q?", [])
    assert r.prompt_tokens == 2 + 900 + 100
    assert r.extra["input_tokens"] == 2 and r.extra["cache_creation_tokens"] == 900 and r.extra["cache_read_tokens"] == 100
    assert r.extra["cost_usd"] == 0.0002 and r.extra["tool_calls"] == 0


def test_more_than_one_turn_or_a_denied_tool_means_tools_were_tried(tmp_path):
    a, _ = make(tmp_path, (0, good(turns=3), ""), (0, good(denials=[{"tool_name": "Read"}]), ""))
    assert a("Q?", []).extra["tool_calls"] == 2
    assert a("Q?", []).extra["tool_calls"] == 1


def test_another_model_answering_is_an_error_not_a_quiet_fallback(tmp_path):
    a, _ = make(tmp_path, (0, good(models={"claude-opus-5-5": {}}), ""))
    with pytest.raises(chat.ChatError):
        a("Q?", [])


@pytest.mark.parametrize("rc,out", [(1, ""), (0, "not json"), (0, good(is_error=True)), (0, json.dumps({"type": "result"}))])
def test_a_failed_or_unreadable_run_is_a_chat_error(tmp_path, rc, out):
    a, _ = make(tmp_path, (rc, out, "boom"))
    with pytest.raises(chat.ChatError):
        a("Q?", [])


def test_the_real_runner_passes_the_input_on_standard_input_and_a_timeout(tmp_path):
    import sys
    rc, out, err = claude_cli.run_process([sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"], "abc", str(tmp_path), 30)
    assert rc == 0 and out.strip() == "ABC"


# ---- step 4: the claude -p child must not update itself in the middle of a run (the version is claimed once) ----

class _Done:
    returncode, stdout, stderr = 0, "{}", ""


def test_the_child_process_gets_the_switch_that_turns_the_auto_updater_off(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        return _Done()
    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)
    monkeypatch.setenv("SOME_OTHER_VARIABLE", "kept")
    claude_cli.run_process(["claude", "-p"], "in", str(tmp_path), 5)
    assert seen["env"]["DISABLE_AUTOUPDATER"] == "1" and seen["env"]["SOME_OTHER_VARIABLE"] == "kept"


def test_the_switch_wins_over_whatever_the_outer_environment_says(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(claude_cli.subprocess, "run", lambda cmd, **kw: (seen.update(kw), _Done())[1])
    monkeypatch.setenv("DISABLE_AUTOUPDATER", "0")
    claude_cli.run_process(["claude", "-p"], "in", str(tmp_path), 5)
    assert seen["env"]["DISABLE_AUTOUPDATER"] == "1"


def test_the_real_answerer_uses_that_runner_and_this_process_environment_is_left_alone(monkeypatch):
    assert claude_cli.ClaudeCliAnswerer(MODEL)._run is claude_cli.run_process
    monkeypatch.delenv("DISABLE_AUTOUPDATER", raising=False)
    monkeypatch.setattr(claude_cli.subprocess, "run", lambda cmd, **kw: _Done())
    claude_cli.run_process(["claude"], "", ".", 5)
    assert "DISABLE_AUTOUPDATER" not in os.environ

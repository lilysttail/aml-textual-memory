# public_gate: install and use

`tools/public_gate.py` is a standard-library-only pre-commit check for repositories that are going to be public.
It blocks private words, local paths, ticket numbers, credential shapes, unknown authors and emails,
unexpected branch and tag names, and a list of known private sentences.

The word list (the "denylist") is private. It is **not** in this repository. Each clone reads it from a local file;
CI reads it from an encrypted secret. If it cannot be read the gate fails with exit code 2; it never skips.

## Install in a clone

```
python tools/install_hooks.py --denylist /path/to/denylist.private.json
```

This sets `core.hooksPath` to `tools/githooks` and remembers the denylist path in the clone's local git config.
Nothing is committed.

## What it checks

| mode | when |
|---|---|
| `--staged` | pre-commit: staged content, paths, and the author about to commit |
| `--commit-msg FILE` | commit-msg hook |
| `--push-stdin` | pre-push hook: target ref names and every commit being pushed |
| `--range A..B`, `--history` | CI and manual audits |
| `--tree [REV]` | the files of one revision |
| `--release` | before the repository is made public |

Exit codes: `0` clean, `1` findings, `2` configuration or environment error.

## Output never repeats what it blocked

Findings are printed from a fixed ASCII template: rule id, object type, commit id, a keyed fingerprint, and
(only when the file path itself is clean) the path and line number. Paths, ref names, authors, emails and
commit messages that triggered a rule are never printed, not even partly. The same template is used for
stdout, stderr, the optional `--report` file and CI annotations.

## CI

`.github/workflows/public-gate.yml` needs two repository secrets:

* `PUBLIC_GATE_DENYLIST`: the denylist file content;
* `PUBLIC_GATE_INTEGRITY`: the output of `python tools/public_gate.py --print-integrity`
  (so that a change to the gate, its rules or its config does not go unnoticed).

Pull requests from forks cannot read secrets, so the job fails for them by design.

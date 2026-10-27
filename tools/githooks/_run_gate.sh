#!/bin/sh
# Shared launcher for the public_gate hooks: finds a working Python and runs tools/public_gate.py.
# Usage (from a hook):  . "$(dirname "$0")/_run_gate.sh"; run_gate --staged
# Output is produced by public_gate.py only; this file never prints repository content.
gate_py=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' >/dev/null 2>&1; then
    gate_py="$c"
    break
  fi
done
if [ -z "$gate_py" ] && command -v py >/dev/null 2>&1; then
  gate_py="py -3"
fi
run_gate() {
  if [ -z "$gate_py" ]; then
    echo "GATE ERROR reason=python_missing" >&2
    return 2
  fi
  root="$(git rev-parse --show-toplevel)" || return 2
  # shellcheck disable=SC2086
  $gate_py "$root/tools/public_gate.py" --repo "$root" "$@"
}

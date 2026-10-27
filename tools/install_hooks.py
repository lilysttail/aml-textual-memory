#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install the public_gate git hooks into the current clone.

    python tools/install_hooks.py [--denylist PATH]

What it does (all local to this clone, nothing is committed):
  * git config core.hooksPath tools/githooks
  * git config publicgate.denylist <PATH>   (only if --denylist or $PUBLIC_GATE_DENYLIST_FILE is given)
  * checks that the denylist can be read and parsed, using the gate itself

It never prints the denylist path or its contents. A missing or unreadable denylist is an error (exit 2),
not a warning: a clone without a readable denylist must not look protected.
"""
import os
import subprocess
import sys


def git(*args):
    return subprocess.run(['git'] + list(args), capture_output=True, text=True, stdin=subprocess.DEVNULL)


def main(argv):
    deny = None
    if '--denylist' in argv:
        i = argv.index('--denylist')
        if i + 1 >= len(argv):
            print('INSTALL ERROR reason=usage', file=sys.stderr)
            return 2
        deny = argv[i + 1]
    deny = deny or os.environ.get('PUBLIC_GATE_DENYLIST_FILE')
    top = git('rev-parse', '--show-toplevel')
    if top.returncode != 0:
        print('INSTALL ERROR reason=not_a_git_repo', file=sys.stderr)
        return 2
    root = top.stdout.strip()
    gate = os.path.join(root, 'tools', 'public_gate.py')
    if not os.path.isfile(gate):
        print('INSTALL ERROR reason=gate_missing', file=sys.stderr)
        return 2
    if git('-C', root, 'config', 'core.hooksPath', 'tools/githooks').returncode != 0:
        print('INSTALL ERROR reason=git_config_failed', file=sys.stderr)
        return 2
    for name in ('pre-commit', 'commit-msg', 'pre-push', '_run_gate.sh'):
        p = os.path.join(root, 'tools', 'githooks', name)
        if not os.path.isfile(p):
            print('INSTALL ERROR reason=hook_missing', file=sys.stderr)
            return 2
        try:
            os.chmod(p, 0o755)
        except OSError:
            pass
        git('-C', root, 'update-index', '--chmod=+x', '--add', os.path.join('tools', 'githooks', name))
    if deny:
        if git('-C', root, 'config', '--local', 'publicgate.denylist', os.path.abspath(deny)).returncode != 0:
            print('INSTALL ERROR reason=git_config_failed', file=sys.stderr)
            return 2
    # Let the gate itself prove the denylist is usable (refs check is cheap and needs no commits).
    probe = subprocess.run([sys.executable, gate, '--repo', root, '--refs'], capture_output=True, text=True,
                           stdin=subprocess.DEVNULL)
    if probe.returncode == 2:
        print('INSTALL ERROR reason=gate_not_usable (%s)' % probe.stderr.strip().replace('GATE ERROR ', ''), file=sys.stderr)
        return 2
    print('INSTALL OK hooks=tools/githooks')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

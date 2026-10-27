#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""public_gate.py — 要公开的仓库的提交前检查。

只用 Python 标准库，单文件。规则本身可以公开；词表（私有）不在这个文件里，也不在公开仓库里：
本地读 --denylist 文件 / 环境变量 / 仓库本地 git config，CI 读加密变量。读不到就报错退出 2，不跳过。

运行方式（互斥）：
  --staged            提交前：暂存区新增内容 + 路径 + 将要提交的作者 / 提交者
  --commit-msg FILE   提交说明文件
  --range SPEC        一段提交（A..B，或单个提交 / 引用）：逐个看说明、作者、新增内容、新增路径
  --history           仓库里全部提交 + 全部引用
  --tree [REV]        某个提交的整棵文件树（默认 HEAD）；--paths A,B 只看这些前缀
  --refs              只检查引用名（分支 / 标签）
  --release           发布前：文件树 + 引用 + 发布标签类型 + LICENSE 版权人 + 占位词
  --push-stdin        pre-push 钩子：从标准输入读 git 给的四列，检查要推的提交和目标引用
  --print-integrity   打印脚本 / 规则 / 配置三个文件的合并 sha256（CI 用它核有没有被改过）

退出码：0 干净；1 有命中；2 配置 / 环境错（词表读不到也是 2）。

输出规矩（3.2a，栖定为 blocker）：门拦住一个东西的时候，自己的输出不能把它再写出来。
  - A 类：内容命中且路径本身干净 → 规则号 + 路径 + 行号 + 指纹；不报那一行的字。
  - B 类：路径 / 引用名 / 作者 / 邮箱 / 提交者 / 提交说明本身命中 → 只报对象类型 + 提交号 + 规则号 + 指纹。
  - 指纹 = HMAC-SHA256(词表里的密钥, 规范化原值) 前 8 位；词表读不到时写 unkeyed，不退回裸哈希。
  - 输出只用固定模板、纯 ASCII；异常只打类型名；输出前再自检一遍，不过就只打一行 OUTPUT_BLOCKED。
"""
import argparse
import fnmatch
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
import unicodedata

EXIT_OK, EXIT_HIT, EXIT_CONFIG = 0, 1, 2
ZERO_SHA = '0' * 40
MAX_FINDINGS_SHOWN = 200
MAX_BLOB_BYTES = 4 * 1024 * 1024
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 内容 / 说明里的放行标记（只对对应的那一类规则生效；词表（G2）没有放行标记）
PROTECT_RULES = frozenset(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G11', 'G12'])
MARKERS = {
    'G1': 'gate:allow-secret',
    'G3': 'gate:allow-path',
    'G4': 'gate:allow-ticket',
    'G6': 'gate:allow-id',
}
CJK_RE = re.compile('[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]')


class GateError(Exception):
    """配置 / 环境错。code 只许是固定的 ASCII 词，绝不带路径或值。"""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


# =========================================================
# 规范化 / 指纹
# =========================================================

def norm_loose(text):
    """原话匹配用：NFKC、小写，去掉空白 / 标点 / 符号 / 控制字符，只留字母数字（换标点、断行都不影响）。"""
    out = []
    for ch in unicodedata.normalize('NFKC', text).lower():
        if ch.isspace() or unicodedata.category(ch)[0] in 'PSZC':
            continue
        out.append(ch)
    return ''.join(out)


def norm_loose_mapped(lines):
    """lines = [(lineno, text)] → (规范化后的整串, 每个字符对应的行号)。"""
    chars, nums = [], []
    for no, text in lines:
        for ch in unicodedata.normalize('NFKC', text).lower():
            if ch.isspace() or unicodedata.category(ch)[0] in 'PSZC':
                continue
            chars.append(ch)
            nums.append(no)
    return ''.join(chars), nums


def safe_ascii(s):
    return all(ord(c) < 128 and (c.isprintable()) for c in s)


# =========================================================
# 词表（私有）
# =========================================================

class Denylist:
    def __init__(self, data):
        if not isinstance(data, dict) or data.get('version') != 1:
            raise GateError('denylist_bad_format')
        key = data.get('key')
        try:
            if not isinstance(key, str) or len(key) < 32:
                raise ValueError
            self.key = bytes.fromhex(key)
        except ValueError:
            raise GateError('denylist_bad_key')
        terms = data.get('terms')
        if not isinstance(terms, list) or not terms:
            raise GateError('denylist_empty')
        self.terms = []
        seen = set()
        for t in terms:
            try:
                tid, kind, val = t['id'], t['kind'], t['value']
                sev = t.get('severity', 'block')
                if not re.fullmatch(r'[A-Za-z0-9_-]{1,32}', tid) or tid in seen or sev not in ('block', 'warn'):
                    raise ValueError
                seen.add(tid)
                if kind == 'literal':
                    self.terms.append({'id': tid, 'kind': kind, 'sev': sev, 'lit': val.lower()})
                elif kind == 'regex':
                    self.terms.append({'id': tid, 'kind': kind, 'sev': sev, 'rx': re.compile(val, re.I | re.A)})
                elif kind == 'quote':
                    nq = norm_loose(val)
                    if len(nq) < 6:
                        raise ValueError
                    self.terms.append({'id': tid, 'kind': kind, 'sev': sev, 'nq': nq})
                else:
                    raise ValueError
            except (KeyError, TypeError, ValueError, re.error):
                raise GateError('denylist_bad_term')
        self.has_quotes = any(t['kind'] == 'quote' for t in self.terms)

    def fp(self, value):
        data = unicodedata.normalize('NFKC', value).strip().lower().encode('utf-8', 'replace')
        return hmac.new(self.key, data, hashlib.sha256).hexdigest()[:8]

    def match_plain(self, text):
        """literal / regex 条目对一段文本（可多行）的命中：[(term_id, severity, 命中的原串)]"""
        hits = []
        low = text.lower()
        for t in self.terms:
            if t['kind'] == 'literal':
                if t['lit'] in low:
                    hits.append((t['id'], t['sev'], t['lit']))
            elif t['kind'] == 'regex':
                m = t['rx'].search(text)
                if m:
                    hits.append((t['id'], t['sev'], m.group(0)))
        return hits

    def match_quotes(self, norm_text):
        """quote 条目对规范化后的整串：[(term_id, severity, 命中起点下标, 命中的规范化串)]"""
        hits = []
        for t in self.terms:
            if t['kind'] == 'quote':
                i = norm_text.find(t['nq'])
                if i >= 0:
                    hits.append((t['id'], t['sev'], i, t['nq']))
        return hits

    def all_values_for_selfcheck(self):
        vals = []
        for t in self.terms:
            if t['kind'] == 'literal':
                vals.append(t['lit'])
            elif t['kind'] == 'quote':
                vals.append(t['nq'])
        return vals


def load_denylist(args, repo):
    raw = None
    if args.denylist:
        path = args.denylist
    else:
        path = os.environ.get('PUBLIC_GATE_DENYLIST_FILE') or None
    if path:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw = f.read()
        except OSError:
            raise GateError('denylist_unreadable')
    elif os.environ.get('PUBLIC_GATE_DENYLIST'):
        raw = os.environ['PUBLIC_GATE_DENYLIST']
    else:
        # 本地钩子装好时把词表路径写进仓库本地 git config（不提交）
        try:
            p = subprocess.run(['git', '-C', repo, 'config', '--get', 'publicgate.denylist'],
                               capture_output=True, text=True, encoding='utf-8', stdin=subprocess.DEVNULL)
            cfgpath = p.stdout.strip() if p.returncode == 0 else ''
        except OSError:
            cfgpath = ''
        if not cfgpath:
            raise GateError('denylist_missing')
        try:
            with open(cfgpath, 'r', encoding='utf-8') as f:
                raw = f.read()
        except OSError:
            raise GateError('denylist_unreadable')
    try:
        data = json.loads(raw)
    except ValueError:
        raise GateError('denylist_bad_format')
    return Denylist(data)


# =========================================================
# 规则 / 配置
# =========================================================

DEFAULT_CONFIG = {
    'version': 1,
    'allowed_identities': [],
    'allowed_refs': {'branches': ['main'], 'tags': ['submission-1']},
    'lightweight_tags': ['submission-1'],
    'release_tag': 'submission-1',
    'allowed_url_prefixes': [],
    'doc_globs': ['*.md', 'LICENSE'],
    'max_file_bytes': 51200,
    'large_file_registry': [],
    'exempt': {},
    'license': {'path': 'LICENSE', 'holder_line': ''},
    'release_placeholders': ['not decided yet', 'to be filled', 'TODO', '(internal)', 'TBD'],
    'third_party': {'dir': '', 'notices': 'THIRD_PARTY_NOTICES.md'},
}


def load_json_file(path, code):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        raise GateError(code)


class Rules:
    def __init__(self, rules_data, cfg):
        self.cfg = cfg
        try:
            self.secret_shapes = []
            for s in rules_data.get('secret_shapes', []):
                try:
                    rx = re.compile(s['regex'], re.A | (re.I if 'i' in s.get('flags', '') else 0))
                except re.error:
                    continue   # JS 专有写法编不了的形状：生成器清单里记了，测试钉住数量
                self.secret_shapes.append((s['name'], rx, int(s.get('valueGroup', 0))))
            w = rules_data['wide_secret']
            self.wide_secret = re.compile(w['regex'], re.I | re.A)
            self.wide_placeholder = re.compile(w['placeholder_regex'], re.I | re.A)
            self.local_path = [re.compile(r, re.A) for r in rules_data['local_path']]
            self.ticket = re.compile(rules_data['ticket'], re.A)
            self.url = re.compile(rules_data['url'], re.A)
            self.notion_ids = [re.compile(r, re.I | re.A) for r in rules_data['notion_ids']]
            self.telegram = [re.compile(r, re.I | re.A) for r in rules_data['telegram']]
            self.email = re.compile(rules_data['email'], re.A)
            self.pseudo_domain = re.compile(rules_data['pseudo_domain'], re.A)
            self.data_deny_globs = list(rules_data['data_deny_globs'])
            self.data_shape = [re.compile(r, re.A) for r in rules_data['data_shape_regexes']]
            self.ai_trailer = [re.compile(r, re.I | re.A) for r in rules_data['ai_trailer']]
            self.provenance = re.compile(rules_data['provenance'], re.I | re.A)
        except (KeyError, TypeError, re.error):
            raise GateError('rules_bad_format')
        self.identities = set()
        for ident in cfg.get('allowed_identities', []):
            self.identities.add((ident.get('name', ''), ident.get('email', '')))

    def exempt(self, rule, path):
        pats = self.cfg.get('exempt', {}).get(rule, [])
        return bool(path) and any(fnmatch.fnmatch(path, p) for p in pats)


def build_config(args, repo):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    path = args.config
    if not path:
        for cand in (os.path.join(repo, 'tools', 'public_gate.config.json'),
                     os.path.join(SCRIPT_DIR, 'public_gate.config.json')):
            if os.path.isfile(cand):
                path = cand
                break
    if path:
        user = load_json_file(path, 'config_unreadable')
        if not isinstance(user, dict):
            raise GateError('config_bad_format')
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg, path


def find_rules_file(args):
    if args.rules:
        return args.rules
    for name in ('public_gate.rules.json', 'rules.default.json'):
        cand = os.path.join(SCRIPT_DIR, name)
        if os.path.isfile(cand):
            return cand
    raise GateError('rules_missing')


# =========================================================
# git
# =========================================================

class Git:
    def __init__(self, repo):
        self.repo = repo

    def run(self, args, input_bytes=None, ok_codes=(0,)):
        try:
            p = subprocess.run(['git', '-c', 'core.quotepath=off', '-C', self.repo] + args,
                               input=input_bytes, capture_output=True,
                               stdin=None if input_bytes is not None else subprocess.DEVNULL)
        except OSError:
            raise GateError('git_unavailable')
        if p.returncode not in ok_codes:
            raise GateError('git_failed')
        return p.stdout

    def text(self, args, **kw):
        return self.run(args, **kw).decode('utf-8', 'replace')

    def rev_list(self, spec_args):
        out = self.text(['rev-list', '--reverse', '--topo-order'] + spec_args)
        return [x for x in out.split('\n') if x]

    def commit_info(self, shas):
        """[(sha, an, ae, cn, ce, message, n_parents)]"""
        if not shas:
            return []
        fmt = '%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%P%x1f%B%x1e'
        out = self.text(['log', '--no-walk=unsorted', '--no-mailmap', '--format=' + fmt] + shas)
        res = []
        for rec in out.split('\x1e'):
            rec = rec.strip('\n')
            if not rec:
                continue
            f = rec.split('\x1f')
            if len(f) < 7:
                continue
            res.append((f[0], f[1], f[2], f[3], f[4], len(f[5].split()), '\x1f'.join(f[6:])))
        return res

    def exists_commit(self, sha):
        try:
            self.run(['cat-file', '-e', sha + '^{commit}'])
            return True
        except GateError:
            return False


def unquote_git_path(p):
    p = p.split('\t')[0]
    if p.startswith('"') and p.endswith('"') and len(p) >= 2:
        body = p[1:-1]
        try:
            return body.encode('latin-1', 'backslashreplace').decode('unicode_escape').encode('latin-1', 'replace').decode('utf-8', 'replace')
        except (UnicodeError, ValueError):
            return body
    return p


def parse_diff_added(text):
    """unified diff（-U0）→ {path: [(新文件行号, 文本)]}"""
    files = {}
    cur = None
    newline = None
    for ln in text.split('\n'):
        if ln.startswith('+++ '):
            p = ln[4:]
            if p.strip() == '/dev/null':
                cur = None
            else:
                p = unquote_git_path(p)
                cur = p[2:] if p.startswith('b/') else p
                files.setdefault(cur, [])
            newline = None
        elif ln.startswith('@@') and cur is not None:
            m = re.match(r'@@ -\S+ \+(\d+)(?:,\d+)? @@', ln)
            newline = int(m.group(1)) if m else None
        elif cur is not None and newline is not None and ln.startswith('+') and not ln.startswith('+++'):
            files[cur].append((newline, ln[1:].rstrip('\r')))
            newline += 1
    return files


# =========================================================
# 检查本体
# =========================================================

class Finding:
    __slots__ = ('rule', 'obj', 'commit', 'path', 'line', 'fp', 'sev', 'why')

    def __init__(self, rule, obj, commit, path, line, fp, sev, why):
        self.rule, self.obj, self.commit, self.path = rule, obj, commit, path
        self.line, self.fp, self.sev, self.why = line, fp, sev, why

    def key(self):
        return (self.rule, self.obj, self.commit or '', self.path or '', self.line or 0, self.fp, self.why or '')


class Gate:
    def __init__(self, git, deny, rules):
        self.git, self.deny, self.rules = git, deny, rules
        self.cfg = rules.cfg
        self.findings = {}
        self.protected = set()       # 自检用：本次检查碰到的、不许出现在输出里的原值（长度 >= 4）
        self._path_cache = {}
        self.stats = {'commits': 0, 'files': 0}

    # ---- 记录 ----
    def _protect(self, value):
        if value and len(value) >= 4 and safe_ascii(value):
            self.protected.add(value.lower())

    def add(self, rule, obj, value='', commit=None, path=None, line=None, sev='block', why=None):
        fp = self.deny.fp(value) if value else 'none'
        if value and rule in PROTECT_RULES:
            self._protect(value)      # G7 / G8 / G9 / G10 的值不是敏感值（常见词、路径），不进自检集合，免得误报
        out_path = None
        if path is not None and obj in ('content', 'file'):
            out_path = self.printable_path(path)
        f = Finding(rule, obj, commit[:12] if commit else None, out_path, line, fp, sev, why)
        self.findings.setdefault(f.key(), f)

    # ---- 路径 ----
    def path_hits(self, path):
        """路径本身的命中：[(rule, value, sev)]（只看路径字符串，不看内容）"""
        if path in self._path_cache:
            return self._path_cache[path]
        hits = []
        for tid, sev, val in self.deny.match_plain(path):
            hits.append(('G2', val, sev))
        if self.deny.has_quotes:
            nq = norm_loose(path)
            for tid, sev, i, val in self.deny.match_quotes(nq):
                hits.append(('G2', val, sev))
        r = self.rules
        if not r.exempt('G6', path):
            for rx in r.notion_ids:
                m = rx.search(path)
                if m:
                    hits.append(('G6', m.group(0), 'block'))
            m = r.email.search(path) or r.pseudo_domain.search(path)
            if m:
                hits.append(('G6', m.group(0), 'block'))
        for rx in r.local_path:
            m = rx.search(path)
            if m and not r.exempt('G3', path):
                hits.append(('G3', m.group(0), 'block'))
        self._path_cache[path] = hits
        return hits

    def printable_path(self, path):
        """A 类才会用到：路径本身有 block 级命中 → 不印（B 类）；非 ASCII → path#指纹；其余原样。"""
        if any(sev == 'block' for _, _, sev in self.path_hits(path)):
            return None
        if safe_ascii(path):
            return path
        return 'path#' + self.deny.fp(path)

    def check_path(self, path, commit=None):
        """路径自身的命中 → B 类 obj=path。返回是否有 block 级命中。"""
        bad = False
        for rule, val, sev in self.path_hits(path):
            self.add(rule, 'path', val, commit=commit, sev=sev)
            bad = bad or sev == 'block'
        return bad

    # ---- 文本（说明 / 内容）----
    def _line_hits(self, line, path, in_message):
        """单行命中：[(rule, 命中串)]（不含词表 quote）"""
        r = self.rules
        hits = []
        # G1 钥匙形状
        if MARKERS['G1'] not in line:
            for name, rx, vg in r.secret_shapes:
                for m in rx.finditer(line):
                    hits.append(('G1', m.group(0)))
            for m in r.wide_secret.finditer(line):
                if not r.wide_placeholder.search(m.group(0)):
                    hits.append(('G1', m.group(0)))
        # G3 本机路径
        if MARKERS['G3'] not in line and not r.exempt('G3', path):
            for rx in r.local_path:
                for m in rx.finditer(line):
                    hits.append(('G3', m.group(0)))
        # G4 楼号（先去掉网址）
        if MARKERS['G4'] not in line and not r.exempt('G4', path):
            stripped = r.url.sub(' ', line)
            for m in r.ticket.finditer(stripped):
                hits.append(('G4', m.group(0)))
        # G6 外部标识
        if MARKERS['G6'] not in line and not r.exempt('G6', path):
            for rx in r.notion_ids:
                for m in rx.finditer(line):
                    hits.append(('G6', m.group(0)))
            for rx in r.telegram:
                for m in rx.finditer(line):
                    hits.append(('G6', m.group(0)))
            for m in r.email.finditer(line):
                hits.append(('G6', m.group(0)))
            for m in r.pseudo_domain.finditer(line):
                hits.append(('G6', m.group(0)))
            prefixes = self.cfg.get('allowed_url_prefixes', [])
            for m in r.url.finditer(line):
                u = m.group(0)
                if not any(u.startswith(p) for p in prefixes):
                    hits.append(('G6', u))
        # G2 词表（literal / regex）。没有放行标记。
        for tid, sev, val in self.deny.match_plain(line):
            hits.append(('G2' if sev == 'block' else 'G2w', val))
        return hits

    def check_content(self, path, lines, commit=None):
        """lines = [(行号, 文本)]。路径本身干净 → A 类（带路径 + 行号）；路径也命中 → 不印路径。"""
        if not lines:
            return
        r = self.rules
        for no, text in lines:
            for rule, val in self._line_hits(text, path, False):
                if rule == 'G2w':
                    self.add('G2', 'content', val, commit=commit, path=path, line=no, sev='warn')
                else:
                    self.add(rule, 'content', val, commit=commit, path=path, line=no)
        # 原话（跨行、跨标点）
        if self.deny.has_quotes:
            norm, nums = norm_loose_mapped(lines)
            for tid, sev, i, val in self.deny.match_quotes(norm):
                self.add('G2', 'content', val, commit=commit, path=path, line=nums[i] if i < len(nums) else None, sev=sev)
        # G8 文档里不许有中日韩字符
        if any(fnmatch.fnmatch(path, g) or fnmatch.fnmatch(os.path.basename(path), g) for g in self.cfg.get('doc_globs', [])):
            for no, text in lines:
                m = CJK_RE.search(text)
                if m:
                    self.add('G8', 'content', m.group(0), commit=commit, path=path, line=no)
        # G7 题库形状（json 里）
        if path.lower().endswith('.json') and not r.exempt('G7', path):
            for no, text in lines:
                for rx in r.data_shape:
                    m = rx.search(text)
                    if m:
                        self.add('G7', 'content', m.group(0), commit=commit, path=path, line=no)
        # G9 出处：写着「源 commit <sha>」但本仓库里没有这个提交
        for no, text in lines:
            for m in r.provenance.finditer(text):
                sha = m.group(1)
                if not self.git.exists_commit(sha):
                    self.add('G9', 'content', sha, commit=commit, path=path, line=no)

    def check_file_meta(self, path, size, commit=None):
        """G7：路径 / 扩展名 / 大小。"""
        r = self.rules
        if r.exempt('G7', path):
            return
        base = os.path.basename(path)
        for g in r.data_deny_globs:
            if fnmatch.fnmatch(path, g) or fnmatch.fnmatch(base, g) or fnmatch.fnmatch(path + '/', g):
                self.add('G7', 'file', g, commit=commit, path=path, why='data_file')
                return
        limit = int(self.cfg.get('max_file_bytes', 51200))
        if size is not None and size > limit and path not in self.cfg.get('large_file_registry', []):
            self.add('G7', 'file', path, commit=commit, path=path, why='too_large')

    def check_message(self, text, commit=None, from_hook=False):
        """提交说明（B 类 obj=message）。from_hook=True 时去掉 git 的 # 注释行。"""
        if from_hook:
            text = '\n'.join(l for l in text.split('\n') if not l.startswith('#'))
        r = self.rules
        lines = [(i + 1, l) for i, l in enumerate(text.split('\n'))]
        for no, l in lines:
            for rule, val in self._line_hits(l, None, True):
                if rule == 'G2w':
                    self.add('G2', 'message', val, commit=commit, sev='warn')
                else:
                    self.add(rule, 'message', val, commit=commit)
            for rx in r.ai_trailer:
                m = rx.search(l)
                if m:
                    self.add('G11', 'message', m.group(0), commit=commit)
        if self.deny.has_quotes:
            norm, _ = norm_loose_mapped(lines)
            for tid, sev, i, val in self.deny.match_quotes(norm):
                self.add('G2', 'message', val, commit=commit, sev=sev)

    def check_identity(self, kind, name, email, commit=None):
        """kind = author / committer。允许清单里的放行；别的拒，再顺带看词表（B 类）。"""
        if (name, email) in self.rules.identities:
            return
        self.add('G5', kind, '%s <%s>' % (name, email), commit=commit, why='not_allowed_identity')
        for part in (name, email):
            for tid, sev, val in self.deny.match_plain(part):
                self.add('G2', kind, val, commit=commit, sev=sev)

    def check_ref(self, name, kind, objtype=None):
        """kind = branch / tag。"""
        allowed = self.cfg['allowed_refs'].get('branches' if kind == 'branch' else 'tags', [])
        if name not in allowed:
            self.add('G12', 'ref', name, why='not_allowed_ref')
        for tid, sev, val in self.deny.match_plain(name):
            self.add('G2', 'ref', val, sev=sev)
        if kind == 'tag' and name in self.cfg.get('lightweight_tags', []) and objtype == 'tag':
            self.add('G12', 'ref', name, why='annotated_tag')

    # ---- 驱动 ----
    def check_refs(self):
        out = self.git.text(['for-each-ref', '--format=%(refname)%09%(objecttype)'])
        for l in out.split('\n'):
            if not l:
                continue
            ref, _, typ = l.partition('\t')
            if ref.startswith('refs/heads/'):
                self.check_ref(ref[len('refs/heads/'):], 'branch')
            elif ref.startswith('refs/tags/'):
                self.check_ref(ref[len('refs/tags/'):], 'tag', typ)
            elif ref.startswith('refs/remotes/'):
                rest = ref[len('refs/remotes/'):]
                name = rest.split('/', 1)[1] if '/' in rest else rest
                if name != 'HEAD':
                    self.check_ref(name, 'branch')

    def check_commits(self, shas):
        for (sha, an, ae, cn, ce, nparents, msg) in self.git.commit_info(shas):
            self.stats['commits'] += 1
            self.check_message(msg, commit=sha)
            self.check_identity('author', an, ae, commit=sha)
            if (cn, ce) != (an, ae):
                self.check_identity('committer', cn, ce, commit=sha)
            if nparents > 1:
                continue    # 合并提交的内容由它的各个父提交覆盖；最终内容由 --tree 看
            diff = self.git.text(['show', '--format=', '--no-color', '--no-ext-diff', '-U0',
                                  '--diff-filter=AMR', '--root', sha])
            added = parse_diff_added(diff)
            for path, lines in added.items():
                self.stats['files'] += 1
                self.check_path(path, commit=sha)
                self.check_file_meta(path, None, commit=sha)
                self.check_content(path, lines, commit=sha)

    def check_staged(self):
        diff = self.git.text(['diff', '--cached', '--no-color', '--no-ext-diff', '-U0', '--diff-filter=AMR'])
        added = parse_diff_added(diff)
        names = self.git.text(['diff', '--cached', '--name-only', '--diff-filter=AMR', '-z']).split('\0')
        for path in [n for n in names if n]:
            self.stats['files'] += 1
            self.check_path(path)
            size = None
            try:
                size = int(self.git.text(['cat-file', '-s', ':' + path]).strip())
            except (GateError, ValueError):
                pass
            self.check_file_meta(path, size)
            self.check_content(path, added.get(path, []))
        for kind, var in (('author', 'GIT_AUTHOR_IDENT'), ('committer', 'GIT_COMMITTER_IDENT')):
            try:
                ident = self.git.text(['var', var]).strip()
            except GateError:
                continue
            m = re.match(r'(.*) <([^>]*)> \d+ [+-]\d{4}$', ident)
            if m:
                self.check_identity(kind, m.group(1), m.group(2))

    def check_tree(self, rev='HEAD', prefixes=None):
        out = self.git.run(['ls-tree', '-r', '-z', '--long', rev])
        entries = []
        for rec in out.split(b'\0'):
            if not rec:
                continue
            meta, _, path = rec.partition(b'\t')
            parts = meta.split()
            if len(parts) < 4 or parts[1] != b'blob':
                continue
            p = path.decode('utf-8', 'replace')
            if prefixes and not any(p == q or p.startswith(q.rstrip('/') + '/') for q in prefixes):
                continue
            size = int(parts[3]) if parts[3].isdigit() else None
            entries.append((parts[2].decode(), p, size))
        for sha, p, size in entries:
            self.stats['files'] += 1
            self.check_path(p)
            self.check_file_meta(p, size)
            if size is not None and size > MAX_BLOB_BYTES:
                continue
            data = self.git.run(['cat-file', 'blob', sha])
            if b'\0' in data[:8192]:
                continue
            text = data.decode('utf-8', 'replace')
            lines = [(i + 1, l.rstrip('\r')) for i, l in enumerate(text.split('\n'))]
            self.check_content(p, lines)
        # 出处声明：third_party 目录下的文件必须在 NOTICES 里有条目
        tp = self.cfg.get('third_party', {})
        if tp.get('dir'):
            try:
                notices = self.git.text(['show', '%s:%s' % (rev, tp.get('notices', 'THIRD_PARTY_NOTICES.md'))])
            except GateError:
                notices = ''
            for sha, p, size in entries:
                if p.startswith(tp['dir']) and os.path.basename(p) not in notices:
                    self.add('G9', 'file', p, path=p, why='no_notice_entry')

    def check_release(self, rev='HEAD'):
        self.check_tree(rev)
        self.check_refs()
        tag = self.cfg.get('release_tag')
        if tag:
            # 标签类型（轻量 / 带说明）由上面的 check_refs 查，这里不再重复，只查发布标签在不在
            try:
                self.git.text(['rev-parse', '--verify', '--quiet', 'refs/tags/' + tag])
            except GateError:
                self.add('G12', 'ref', tag, why='missing_release_tag')
        lic = self.cfg.get('license', {})
        holder = lic.get('holder_line')
        if holder:
            try:
                body = self.git.text(['show', '%s:%s' % (rev, lic.get('path', 'LICENSE'))])
                if holder not in body.split('\n'):
                    self.add('G10', 'file', holder, path=lic.get('path', 'LICENSE'), why='license_holder')
            except GateError:
                self.add('G10', 'file', holder, path=lic.get('path', 'LICENSE'), why='missing_license')
        holders = [h.lower() for h in self.cfg.get('release_placeholders', [])]
        for sha, p, _ in self._tree_blobs(rev):
            if any(fnmatch.fnmatch(p, g) or fnmatch.fnmatch(os.path.basename(p), g) for g in self.cfg.get('doc_globs', [])):
                data = self.git.run(['cat-file', 'blob', sha]).decode('utf-8', 'replace')
                for i, l in enumerate(data.split('\n')):
                    low = l.lower()
                    for h in holders:
                        if h in low:
                            self.add('G10', 'content', h, path=p, line=i + 1, why='placeholder')

    def _tree_blobs(self, rev):
        out = self.git.run(['ls-tree', '-r', '-z', '--long', rev])
        res = []
        for rec in out.split(b'\0'):
            if not rec:
                continue
            meta, _, path = rec.partition(b'\t')
            parts = meta.split()
            if len(parts) >= 4 and parts[1] == b'blob':
                res.append((parts[2].decode(), path.decode('utf-8', 'replace'), None))
        return res

    def check_push_stdin(self, stdin_text):
        """pre-push：每行 `local_ref local_sha remote_ref remote_sha`。"""
        for l in stdin_text.split('\n'):
            parts = l.split()
            if len(parts) != 4:
                continue
            lref, lsha, rref, rsha = parts
            if rref.startswith('refs/heads/'):
                self.check_ref(rref[len('refs/heads/'):], 'branch')
            elif rref.startswith('refs/tags/'):
                typ = ''
                if lsha != ZERO_SHA:
                    try:
                        typ = self.git.text(['cat-file', '-t', lsha]).strip()
                    except GateError:
                        typ = ''
                self.check_ref(rref[len('refs/tags/'):], 'tag', typ)
            if lsha == ZERO_SHA:
                continue    # 删除
            if rsha == ZERO_SHA:
                shas = self.git.rev_list([lsha, '--not', '--remotes'])
            else:
                shas = self.git.rev_list([rsha + '..' + lsha])
            self.check_commits(shas)


# =========================================================
# 输出（3.2a）
# =========================================================

def render(gate, mode, limit=MAX_FINDINGS_SHOWN):
    fails = sorted([f for f in gate.findings.values() if f.sev == 'block'], key=lambda f: f.key())
    warns = sorted([f for f in gate.findings.values() if f.sev == 'warn'], key=lambda f: f.key())
    lines = []

    def fmt(tag, f):
        parts = ['GATE ' + tag, 'rule=' + f.rule, 'obj=' + f.obj]
        if f.why:
            parts.append('why=' + f.why)
        if f.commit:
            parts.append('commit=' + f.commit)
        if f.path:
            parts.append('path=' + f.path)
        if f.line:
            parts.append('line=%d' % f.line)
        parts.append('fp=' + f.fp)
        return ' '.join(parts)

    for f in fails[:limit]:
        lines.append(fmt('FAIL', f))
    if len(fails) > limit:
        lines.append('GATE TRUNCATED more=%d' % (len(fails) - limit))
    for f in warns[:limit]:
        lines.append(fmt('WARN', f))
    status = 'FAIL' if fails else 'PASS'
    lines.append('GATE RESULT status=%s mode=%s findings=%d warnings=%d commits=%d files=%d'
                 % (status, mode, len(fails), len(warns), gate.stats['commits'], gate.stats['files']))
    return lines, bool(fails)


def self_check(lines, gate):
    """输出前兜底：整份输出必须纯 ASCII 可打印，且不含任何被保护的原值 / 词表里的字面词。"""
    text = '\n'.join(lines)
    if not all(ord(c) < 128 and (c.isprintable() or c == '\n') for c in text):
        return False
    low = text.lower()
    for v in gate.protected:
        if v in low:
            return False
    for v in gate.deny.all_values_for_selfcheck():
        if len(v) >= 4 and safe_ascii(v) and v in low:
            return False
    return True


def annotation(line):
    return '::error title=public-gate::' + line


def emit(lines, args, failed):
    out = list(lines)
    if args.report:
        try:
            with open(args.report, 'w', encoding='ascii', newline='\n') as f:
                f.write('\n'.join(out) + '\n')
        except OSError:
            raise GateError('report_unwritable')
    sys.stdout.write('\n'.join(out) + '\n')
    if args.annotations:
        for l in out:
            if l.startswith('GATE FAIL') or l.startswith('GATE TRUNCATED'):
                sys.stdout.write(annotation(l) + '\n')
    sys.stdout.flush()


def emit_error(code, args=None):
    line = 'GATE ERROR reason=%s' % code
    sys.stderr.write(line + '\n')
    sys.stderr.flush()


# =========================================================
# 入口
# =========================================================

class QuietParser(argparse.ArgumentParser):
    def error(self, message):     # argparse 默认会把传进来的值回显到 stderr
        raise GateError('usage')

    def exit(self, status=0, message=None):
        raise GateError('usage')


def build_parser():
    p = QuietParser(prog='public_gate.py', add_help=False)
    p.add_argument('--repo', default='.')
    p.add_argument('--config')
    p.add_argument('--rules')
    p.add_argument('--denylist')
    p.add_argument('--report')
    p.add_argument('--annotations', action='store_true')
    p.add_argument('--paths')
    p.add_argument('--max-shown', dest='max_shown', type=int, default=MAX_FINDINGS_SHOWN)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--staged', action='store_true')
    g.add_argument('--commit-msg', dest='commit_msg')
    g.add_argument('--range', dest='range_spec')
    g.add_argument('--history', action='store_true')
    g.add_argument('--tree', nargs='?', const='HEAD')
    g.add_argument('--refs', action='store_true')
    g.add_argument('--release', action='store_true')
    g.add_argument('--push-stdin', dest='push_stdin', action='store_true')
    g.add_argument('--print-integrity', dest='print_integrity', action='store_true')
    return p


def integrity(rules_path, cfg_path):
    h = hashlib.sha256()
    for path in (os.path.abspath(__file__), rules_path, cfg_path or ''):
        h.update(b'--file--')
        if path and os.path.isfile(path):
            with open(path, 'rb') as f:
                h.update(f.read().replace(b'\r\n', b'\n'))
    return h.hexdigest()


def main(argv=None):
    args = None
    try:
        args = build_parser().parse_args(argv)
        repo = os.path.abspath(args.repo)
        rules_path = find_rules_file(args)
        cfg, cfg_path = build_config(args, repo)
        if args.print_integrity:
            sys.stdout.write(integrity(rules_path, cfg_path) + '\n')
            return EXIT_OK
        deny = load_denylist(args, repo)
        rules = Rules(load_json_file(rules_path, 'rules_unreadable'), cfg)
        git = Git(repo)
        gate = Gate(git, deny, rules)
        mode = None
        if args.staged:
            mode = 'staged'
            gate.check_staged()
        elif args.commit_msg:
            mode = 'commit-msg'
            try:
                with open(args.commit_msg, 'rb') as f:
                    gate.check_message(f.read().decode('utf-8', 'replace'), from_hook=True)
            except OSError:
                raise GateError('message_unreadable')
        elif args.range_spec:
            mode = 'range'
            spec = args.range_spec
            gate.check_commits(git.rev_list([spec]))
        elif args.history:
            mode = 'history'
            gate.check_commits(git.rev_list(['--all']))
            gate.check_refs()
        elif args.tree is not None:
            mode = 'tree'
            prefixes = [x for x in (args.paths or '').split(',') if x] or None
            gate.check_tree(args.tree, prefixes)
        elif args.refs:
            mode = 'refs'
            gate.check_refs()
        elif args.release:
            mode = 'release'
            gate.check_release('HEAD')
        elif args.push_stdin:
            mode = 'push'
            gate.check_push_stdin(sys.stdin.read())
        lines, failed = render(gate, mode, max(1, args.max_shown))
        if not self_check(lines, gate):
            blocked = ['GATE FAIL rule=O1 obj=output fp=none',
                       'GATE RESULT status=FAIL mode=%s findings=1 warnings=0 commits=0 files=0' % mode]
            emit(blocked, args, True)
            return EXIT_HIT
        emit(lines, args, failed)
        return EXIT_HIT if failed else EXIT_OK
    except GateError as e:
        emit_error(e.code, args)
        return EXIT_CONFIG
    except SystemExit:
        raise
    except BaseException as e:    # noqa: B902 —— 任何异常只打类型名，不打消息（库的异常消息常带路径和值）
        sys.stderr.write('GATE ERROR reason=internal type=%s\n' % ''.join(c for c in type(e).__name__ if c.isalnum()))
        sys.stderr.flush()
        return EXIT_CONFIG


if __name__ == '__main__':
    sys.exit(main())

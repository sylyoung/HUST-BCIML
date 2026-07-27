#!/usr/bin/env python3
"""Check the web app's Chinese layer against the conventions it is held to.

Four failure modes this catches, all of which shipped at least once:

1. **Key drift.** ``app.js`` calls ``tr("<English>")`` and ``i18n.js`` maps the
   *exact* English string to its translation. Editing the English in app.js
   without updating the key silently falls back to English — the string is still
   there, still renders, and only a reader of the Chinese page notices. The whole
   leaderboard reading guide was English in the zh view this way.
2. **Untranslated content strings.** The longest prose on the site — table titles,
   their blurbs, the sub-category headers, the dataset roles — is not written in
   ``app.js`` at all. It comes from ``gallery/data/benchmark.yml`` through the build,
   and the app translates it with the same ``tr()``, so it needs an ``i18n.js`` key
   just as much. Checking only the literal ``tr("…")`` calls left every one of those
   unchecked: a new leaderboard table would render its title and its whole
   explanatory blurb in English on the Chinese page, with nothing failing.
3. **Forbidden punctuation.** Em-dashes, curly quotes and semicolons in Chinese
   prose are the strongest machine-translation tell, and are not used here.
4. **Banned renderings.** 伍老师 for the professor, and calques such as 陈列 /
   事实来源 / 旗舰 that a translator reaches for and a reader trips over.

#3 and #4 are conventions about Chinese prose, not about the web app, so they are
applied to ``README.zh-CN.md`` as well. It is the repo's other Chinese artifact and
the first one most readers meet, and it was held to these rules by nothing at all —
the rules lived only in a check that reads ``i18n.js``.

Because #2 reads the *built* ``docs/data/benchmark.js``, run the build first — a
stale build would check yesterday's strings.

Run: python3 gallery/check_i18n.py     (from the repo root; exit 1 on any finding)
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
APP = os.path.join(ROOT, "docs", "assets", "app.js")
I18N = os.path.join(ROOT, "docs", "assets", "i18n.js")
BENCH_JS = os.path.join(ROOT, "docs", "data", "benchmark.js")
LAB_JS = os.path.join(ROOT, "docs", "data", "lab.js")
PUBS_JS = os.path.join(ROOT, "docs", "data", "publications.js")
README_EN = os.path.join(ROOT, "README.md")
README_ZH = os.path.join(ROOT, "README.zh-CN.md")

BAD_PUNCT = {"——": "em-dash", "；": "full-width semicolon",
             "“": "curly open quote", "”": "curly close quote"}
BANNED = ("伍老师", "陈列", "事实来源", "旗舰")

# A tr() argument may be several adjacent string literals joined by +, split over
# lines for readability. At runtime tr() receives the concatenation, so the key to
# look up is the concatenation — checking only the first literal reports false
# drift on every wrapped string.
TR_CALL = re.compile(r'tr\(\s*((?:"(?:[^"\\]|\\.)*"\s*\+\s*)*"(?:[^"\\]|\\.)*")')
LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')
KEY_LINE = re.compile(r'^\s{6}"((?:[^"\\]|\\.)*)"\s*:', re.M)
ZH_VALUE = re.compile(r'^\s{8}"((?:[^"\\]|\\.)*)"\s*,?\s*$', re.M)


def unescape(s):
    return s.replace('\\"', '"').replace("\\\\", "\\")


def _embedded(path, open_ch, close_ch):
    """The JSON value out of a `window.X = {...};` data file."""
    raw = open(path, encoding="utf-8").read()
    return json.loads(raw[raw.index(open_ch):raw.rindex(close_ch) + 1])


def content_strings():
    """[(where, English)] for every generated string the app runs through tr().

    Mirrors the ``tr(t.title)`` / ``tr(t.blurb)`` / ``tr(g.subcat)`` / ``tr(g.blurb)``
    / ``tr(d.role)`` calls in app.js, and the three data files they read from — the
    benchmark, the lab card, and the publications' research areas. Row names and method
    descriptions are deliberately absent: those render untranslated by design (a
    method's name is its identifier, and the per-row description sits behind the card
    link).

    Covering only ``benchmark.js`` left a live gap for as long as this check existed:
    a new research area on the publications page, or an edited tagline on the lab card,
    goes through ``tr()`` exactly like a table blurb and would have rendered in English
    on the Chinese page with nothing to say so.
    """
    if not os.path.exists(BENCH_JS):
        return None                                  # unbuilt tree; reported by caller
    bench = _embedded(BENCH_JS, "{", "}")
    out = []
    for t in bench.get("tables") or []:
        out.append((f"table {t.get('id')!r} title", t.get("title")))
        out.append((f"table {t.get('id')!r} blurb", t.get("blurb")))
        for g in t.get("groups") or []:
            out.append((f"table {t.get('id')!r} subcat", g.get("subcat")))
            out.append((f"table {t.get('id')!r} subcat blurb", g.get("blurb")))
    for d in (bench.get("meta") or {}).get("datasets") or []:
        out.append((f"dataset {d.get('name')!r} role", d.get("role")))
        # `trials` is "288 / session" for a dataset counted per session and a bare
        # number otherwise. Only the wordy form needs a translation, and requiring one
        # for "100" would be noise — so the test is whether the value contains a letter.
        trials = d.get("trials")
        if trials is not None and re.search(r"[A-Za-z]", str(trials)):
            out.append((f"dataset {d.get('name')!r} trials", str(trials)))

    if os.path.exists(LAB_JS):
        # lab.js holds two objects; the leading one is window.LAB.
        raw = open(LAB_JS, encoding="utf-8").read()
        lab = json.loads(raw[raw.index("{"):raw.index("\n};") + 2])
        lib = lab.get("library") or {}
        for field in ("title", "tagline", "driver"):        # tr(lib.*) in app.js
            out.append((f"lab card {field}", lib.get(field)))
        for group in ("mounts", "channels", "links", "official"):
            for m in lab.get(group) or []:
                if isinstance(m, dict):
                    out.append((f"lab {group} entry", m.get("name")))  # tr(mnt.name)

    if os.path.exists(PUBS_JS):
        pubs = _embedded(PUBS_JS, "[", "]")
        for topic in sorted({p.get("topic") for p in pubs if p.get("topic")}):
            out.append((f"research area {topic!r}", topic))   # tr(p.topic)

    return [(w, s) for w, s in out if s]


def main():
    app = open(APP, encoding="utf-8").read()
    i18n = open(I18N, encoding="utf-8").read()
    findings = []

    keys = {unescape(k) for k in KEY_LINE.findall(i18n)}
    calls = {unescape("".join(LITERAL.findall(arg))) for arg in TR_CALL.findall(app)}

    for missing in sorted(calls - keys):
        findings.append(f"tr() string has no i18n key (renders English in zh):\n"
                        f"    {missing[:160]}{'...' if len(missing) > 160 else ''}")

    content = content_strings()
    if content is None:
        findings.append(f"no built {os.path.relpath(BENCH_JS, ROOT)} — run "
                        f"gallery/build_site.py first, or the benchmark's own prose "
                        f"goes unchecked")
        n_content = 0
    else:
        n_content = len(content)
        for where, s in content:
            if s not in keys:
                findings.append(f"a generated string has no i18n key ({where} "
                                f"renders English in zh):\n"
                                f"    {s[:160]}{'...' if len(s) > 160 else ''}")

    # Every quoted run on a value line; the zh side is what the reader sees.
    for line_no, line in enumerate(i18n.splitlines(), 1):
        m = ZH_VALUE.match(line)
        if not m:
            continue
        val = unescape(m.group(1))
        for bad, name in BAD_PUNCT.items():
            if bad in val:
                findings.append(f"i18n.js:{line_no} contains a {name} ({bad}): "
                                f"{val[:80]}...")
        for word in BANNED:
            if word in val:
                findings.append(f"i18n.js:{line_no} uses banned rendering {word}: "
                                f"{val[:80]}...")

    # The same two prose conventions, on the Chinese README. Fenced code and inline
    # code are skipped: a semicolon inside a shell command is a shell semicolon, and
    # flagging it would train the reader to ignore this check.
    n_zh_lines = 0
    if os.path.exists(README_ZH):
        in_fence = False
        for line_no, line in enumerate(open(README_ZH, encoding="utf-8"), 1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            prose = re.sub(r"`[^`]*`", "", line)
            n_zh_lines += 1
            for bad, name in BAD_PUNCT.items():
                if bad in prose:
                    findings.append(f"README.zh-CN.md:{line_no} contains a {name} "
                                    f"({bad}): {prose.strip()[:80]}...")
            for word in BANNED:
                if word in prose:
                    findings.append(f"README.zh-CN.md:{line_no} uses banned rendering "
                                    f"{word}: {prose.strip()[:80]}...")

    # The two READMEs must have the same section skeleton. Nothing else compares them,
    # and the failure is silent by construction: the English file gains a section and
    # the Chinese one keeps rendering perfectly without it. That is how README.zh-CN.md
    # came to state, four months after the English text stopped saying it, that every
    # hyperparameter was selected on source subjects alone — a measurement-integrity
    # claim that had been retracted in English and left standing in Chinese. Heading
    # levels are compared, not their text, which differs by design.
    shapes = {}
    for label, path in (("README.md", README_EN), ("README.zh-CN.md", README_ZH)):
        if not os.path.exists(path):
            continue
        levels, in_fence = [], False
        for line_no, line in enumerate(open(path, encoding="utf-8"), 1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            m = re.match(r"(#{1,6})\s+\S", line)
            if m and not in_fence:                  # a `#` inside a fence is a comment
                levels.append((len(m.group(1)), line_no, line.strip()))
        shapes[label] = levels
    if len(shapes) == 2:
        en, zh = shapes["README.md"], shapes["README.zh-CN.md"]
        # A level mismatch inside the common prefix pins the divergence exactly — that
        # is the nesting case, one language gaining a subsection. A file that is merely
        # shorter does not: dropping a `##` leaves every later level unchanged, so the
        # first difference is at the end no matter where the section was removed.
        # Pointing there would send the reader to the wrong place, so say which it is.
        first = next((i for i in range(min(len(en), len(zh))) if en[i][0] != zh[i][0]), None)
        if first is not None:
            a, b = en[first], zh[first]
            findings.append(
                f"the READMEs diverge at heading {first + 1}: "
                f"README.md:{a[1]} {a[2][:56]!r} (level {a[0]}) vs "
                f"README.zh-CN.md:{b[1]} {b[2][:56]!r} (level {b[0]})")
        elif len(en) != len(zh):
            longer, name, other = ((en, "README.md", "README.zh-CN.md") if len(en) > len(zh)
                                   else (zh, "README.zh-CN.md", "README.md"))
            extra = longer[min(len(en), len(zh)):]
            findings.append(
                f"{name} has {len(extra)} heading(s) that {other} does not, and the two "
                f"agree on every level before them, so the missing section is not "
                f"necessarily the last one. {name} ends with: "
                + ", ".join(f"{h[2][:40]!r} (line {h[1]})" for h in extra))

    if findings:
        print(f"{len(findings)} finding(s):\n")
        for f in findings:
            print(" -", f)
        return 1
    print(f"i18n OK — {len(calls)} tr() strings in app.js and {n_content} generated "
          f"content strings (benchmark, lab card, research areas) all resolve, "
          f"{len(keys)} keys defined, the two READMEs have the same section skeleton, "
          f"and there is no forbidden punctuation or rendering in i18n.js or across "
          f"{n_zh_lines} prose lines of README.zh-CN.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

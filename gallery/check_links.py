#!/usr/bin/env python3
# check_links.py  —  HUST-BCIML paper-to-code gallery.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Resolve every paper and code link in the gallery, and report rot.

The gallery's whole premise is that each paper is linked to its released code.
That premise decays quietly: the page keeps rendering a dead link exactly as it
renders a live one. And the code lives across dozens of student GitHub accounts,
any of which can rename or delete a repository at any time — a rename leaves a
301 that works only for as long as GitHub keeps the alias, which ends the moment
someone re-registers the old name.

Checked, from ``data/publications.yml`` and ``data/benchmark.yml``:

* ``code_url``  — HTTP HEAD (falling back to GET for hosts that reject HEAD).
* ``paper_url`` — same.
* ``doi``       — resolved through ``https://doi.org/<doi>``.

Exit status: non-zero if anything is dead (4xx/5xx or unreachable). Redirects are
reported as warnings with their target, because the fix is to store the new URL,
not to treat the redirect as failure — except for a ``doi.org`` link, where the
redirect *is* the resolution and storing the target would throw away the permanent
identifier. Publisher 403s (Cloudflare, paywalls) are warnings too — they mean
"the checker was blocked", not "the paper is gone".

    python gallery/check_links.py                 # everything
    python gallery/check_links.py --code-only
    python gallery/check_links.py --timeout 20
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# Statuses that mean "the checker was refused", not "the target is gone".
BLOCKED = {401, 403, 429}

DOI_BASE = "https://doi.org/"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Capture the first redirect instead of following it, so a rename is visible."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise _Redirected(code, newurl)


class _Redirected(Exception):
    def __init__(self, code, target):
        super().__init__(f"{code} -> {target}")
        self.code, self.target = code, target


def check(url: str, timeout: float):
    """Return ``(status, detail)`` where status is ok / redirect / blocked / dead."""
    opener = urllib.request.build_opener(_NoRedirect)
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, method=method, headers={"User-Agent": UA})
        try:
            with opener.open(req, timeout=timeout) as resp:
                return "ok", str(resp.status)
        except _Redirected as r:
            return "redirect", r.target
        except urllib.error.HTTPError as e:
            if e.code in BLOCKED:
                return "blocked", f"HTTP {e.code} (checker refused, not necessarily dead)"
            if method == "HEAD" and e.code in (405, 501):
                continue                          # host rejects HEAD; retry with GET
            return "dead", f"HTTP {e.code}"
        except Exception as e:                    # DNS, TLS, timeout, connection reset
            if method == "HEAD":
                continue
            return "dead", f"{type(e).__name__}: {e}"
    return "dead", "no response to HEAD or GET"


def collect(code_only: bool):
    """Every (label, url) pair worth checking, de-duplicated, in a stable order."""
    seen, out = set(), []

    def add(label, url):
        if url and url not in seen:
            seen.add(url)
            out.append((label, url))

    pubs_path = os.path.join(DATA, "publications.yml")
    with open(pubs_path) as fh:
        pubs = yaml.safe_load(fh) or {}
    for p in (pubs.get("publications") if isinstance(pubs, dict) else pubs) or []:
        pid = p.get("id") or p.get("title", "")[:40]
        add(f"code   {pid}", p.get("code_url"))
        if not code_only:
            add(f"paper  {pid}", p.get("paper_url"))
            if p.get("doi"):
                # An in-press paper has a DOI the publisher has assigned but not yet
                # registered, so it 404s until the issue appears. Reporting that as
                # rot trains the reader to ignore the dead list, which is the one
                # thing this checker cannot afford. Check the parent instead: an ACM
                # article DOI is <proceedings>.<article>, and the proceedings DOI is
                # registered when the volume is. That still proves the venue exists,
                # so the entry is verified rather than merely trusted.
                if p.get("in_press"):
                    parent = p["doi"].rsplit(".", 1)[0]
                    if "/" in parent and parent != p["doi"]:
                        add(f"venue  {pid} (in press, parent DOI)",
                            DOI_BASE + parent)
                else:
                    add(f"doi    {pid}", DOI_BASE + p["doi"])

    bm_path = os.path.join(DATA, "benchmark.yml")
    if os.path.exists(bm_path) and not code_only:
        with open(bm_path) as fh:
            bm = yaml.safe_load(fh) or {}
        for t in bm.get("tables", []):
            for g in (t.get("groups") or [t]):
                rows = list(g.get("rows") or [])
                # A group's `reference` renders as a row and carries its own DOI, so
                # leaving it out meant every baseline row's paper link went unchecked —
                # the rows every other row in the table is measured against.
                if g.get("reference"):
                    rows.append(g["reference"])
                for r in rows:
                    if r.get("doi"):
                        add(f"doi    {r.get('key') or r.get('name')}",
                            DOI_BASE + r["doi"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="check_links")
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--code-only", action="store_true",
                    help="check only code_url (fast; the links most likely to rot)")
    a = ap.parse_args(argv)

    targets = collect(a.code_only)
    print(f"checking {len(targets)} links\n")
    dead, redirects, blocked, resolved = [], [], [], 0
    for label, url in targets:
        status, detail = check(url, a.timeout)
        # A DOI redirecting to a publisher is the DOI doing its job — indirection is
        # the entire point of the identifier — so this fires for every registered DOI
        # and separates nothing. It fired 287 times on 287 DOIs here, burying the
        # question the run exists to answer: did any *code* link move? And the advice
        # attached to a redirect, store the target instead, is actively wrong for a
        # DOI: it would swap the one permanent URL for the publisher URL it exists to
        # insulate the gallery against. A DOI that has really rotted does not redirect,
        # it 404s at doi.org, and lands in `dead` regardless. So resolving counts as
        # ok, and `redirect` keeps its meaning for code and paper URLs, where a 301 is
        # a rename someone has to act on.
        if status == "redirect" and url.startswith(DOI_BASE):
            resolved += 1
            continue
        if status == "ok":
            continue
        line = f"{label}\n    {url}\n    -> {detail}"
        {"dead": dead, "redirect": redirects, "blocked": blocked}[status].append(line)
        print(f"[{status:8s}] {label}: {url} -> {detail}", flush=True)

    print(f"\n{len(targets) - len(dead) - len(redirects) - len(blocked)} ok "
          f"({resolved} of them DOIs that resolved to a publisher), "
          f"{len(redirects)} redirected, {len(blocked)} blocked, {len(dead)} dead")
    if redirects:
        print("\nRedirected (update the stored URL — the alias only lives as long as the "
              "old name stays unregistered):")
        print("\n".join(redirects))
    if dead:
        print("\nDead:")
        print("\n".join(dead))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

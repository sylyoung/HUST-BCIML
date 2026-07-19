/* HUST BCIML web app. Vanilla JS, no framework.
   Data comes from window.LAB / window.PUBLICATIONS / window.BENCHMARK (inlined).
   Code-first: the repo's own assets (benchmark + code) lead; the lab's profile and
   full publication list live on the official sites linked in the Overview. */
(function () {
  "use strict";
  var LAB = window.LAB || {}, SITE = window.SITE || {};
  var PUBS = window.PUBLICATIONS || [], BENCH = window.BENCHMARK || {};
  var VIEWS = ["overview", "benchmark", "papers"];
  var REPO_URL = "https://github.com/sylyoung/HUST-BCIML";
  var PKG = "hustbciml";   // the benchmark package directory in the repo

  // ---------------- i18n ----------------
  // LANG is 'en' or 'zh'. tr(s) returns the Chinese translation of the EXACT
  // English source string s when in Chinese mode and a mapping exists; otherwise
  // it returns s unchanged. Keep-English fields (paper titles, method desc/ref,
  // per-repo blurbs, names, URLs, DOIs) are never passed through tr().
  var LANG = (function () { try { return localStorage.getItem("lang") || "en"; } catch (e) { return "en"; } })();
  var I18N = window.I18N || {};
  function tr(s) {
    if (LANG === "zh" && I18N.zh && Object.prototype.hasOwnProperty.call(I18N.zh, s)) return I18N.zh[s];
    return s;
  }
  // dataset column order for the three-dataset leaderboard tables
  var DS = (BENCH.datasets && BENCH.datasets.length) ? BENCH.datasets
    : ((BENCH.meta && BENCH.meta.datasets) || []).map(function (d) { return d.name; });

  // ---------------- DOM helper ----------------
  function el(tag, attrs, children) {
    var n = document.createElement(tag);
    if (attrs) {
      for (var k in attrs) {
        var v = attrs[k];
        if (v == null) continue;
        if (k === "class") n.className = v;
        else if (k.slice(0, 2) === "on" && typeof v === "function") n.addEventListener(k.slice(2), v);
        else n.setAttribute(k, v);
      }
    }
    if (children != null) {
      (Array.isArray(children) ? children : [children]).forEach(function (c) {
        if (c == null) return;
        n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
      });
    }
    return n;
  }
  function txt(s) { return document.createTextNode(s); }
  function hue(s) { var h = 0; for (var i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360; return h; }
  function counts(field, isArray) {
    var m = new Map();
    PUBS.forEach(function (p) {
      var vals = isArray ? (p[field] || []) : [p[field]];
      vals.forEach(function (v) { if (v == null || v === "") return; m.set(v, (m.get(v) || 0) + 1); });
    });
    return Array.from(m.entries()).sort(function (a, b) { return b[1] - a[1]; });
  }

  var topicList = counts("topic", false);
  var paradigmList = counts("paradigm", true);

  // ---------------- navigation ----------------
  function activate(name) {
    if (VIEWS.indexOf(name) < 0) name = "overview";
    document.querySelectorAll(".tab").forEach(function (t) { t.classList.toggle("active", t.dataset.view === name); });
    document.querySelectorAll(".view").forEach(function (v) { v.classList.toggle("active", v.id === name); });
  }
  function go(name) { activate(name); history.replaceState(null, "", "#" + name); window.scrollTo({ top: 0 }); }
  document.getElementById("tabs").addEventListener("click", function (e) {
    // only view tabs carry data-view; the language toggle (also .tab) is skipped here
    var b = e.target.closest(".tab"); if (b && b.dataset.view) go(b.dataset.view);
  });
  document.getElementById("brand").addEventListener("click", function (e) { e.preventDefault(); go("overview"); });
  window.addEventListener("hashchange", function () { activate(location.hash.slice(1)); window.scrollTo({ top: 0 }); });

  // ---------------- header / footer ----------------
  function header() {
    var hl = document.getElementById("header-links"), L = LAB.links || {};
    hl.textContent = "";
    if (L.lab_site) hl.appendChild(el("a", { href: L.lab_site, target: "_blank", rel: "noopener" }, tr("Lab site")));
    if (L.pi_homepage) hl.appendChild(el("a", { href: L.pi_homepage, target: "_blank", rel: "noopener" }, tr("Prof. Wu")));
    if (L.scholar) hl.appendChild(el("a", { href: L.scholar, target: "_blank", rel: "noopener" }, tr("Scholar")));
    hl.appendChild(el("a", { href: REPO_URL, target: "_blank", rel: "noopener" }, tr("GitHub")));
    var mnt = LAB.maintainer || {};
    if (mnt.homepage) hl.appendChild(el("a", { href: mnt.homepage, target: "_blank", rel: "noopener", title: tr("Maintainer") }, mnt.name ? tr(mnt.name) : tr("Maintainer")));
  }
  function footer() {
    var f = document.getElementById("footer"), L = LAB.links || {}, mnt = LAB.maintainer || {};
    f.textContent = "";
    var l1 = el("div", { class: "foot-line" });
    l1.appendChild(txt(tr(LAB.full_name || "HUST BCIML") + " · " + tr(LAB.pi || "") + " · "));
    if (L.lab_site) { l1.appendChild(el("a", { href: L.lab_site, target: "_blank", rel: "noopener" }, tr("Lab website"))); l1.appendChild(txt(" · ")); }
    if (L.pi_homepage) { l1.appendChild(el("a", { href: L.pi_homepage, target: "_blank", rel: "noopener" }, tr("Prof. Wu's homepage"))); l1.appendChild(txt(" · ")); }
    l1.appendChild(el("a", { href: REPO_URL, target: "_blank", rel: "noopener" }, tr("Repository")));
    f.appendChild(l1);

    var l2 = el("div", { class: "foot-line" });
    l2.appendChild(txt(tr("Benchmark and web app built and maintained by ")));
    if (mnt.homepage) l2.appendChild(el("a", { href: mnt.homepage, target: "_blank", rel: "noopener" }, tr(mnt.name || "Siyang Li")));
    else l2.appendChild(el("strong", {}, tr(mnt.name || "Siyang Li")));
    if (mnt.email) { l2.appendChild(txt(" · ")); l2.appendChild(el("a", { href: "mailto:" + mnt.email }, mnt.email)); }
    l2.appendChild(txt(tr(". Prof. Wu's email is available in any of the lab's publications.")));
    f.appendChild(l2);

    f.appendChild(el("div", { class: "foot-disclaimer" },
      tr("Disclaimer: this benchmark reimplements both external baselines and the lab's own methods independently. The reported numbers — baseline reproductions and lab-method results alike — may differ from the original papers and can contain errors. Corrections are welcome; please contact the maintainer.")));
  }

  // ---------------- overview ----------------
  function stat(parent, n, label) {
    var s = el("div", { class: "stat" });
    s.appendChild(el("div", { class: "n" }, n == null ? "—" : String(n)));
    s.appendChild(el("div", { class: "l" }, label));
    parent.appendChild(s);
  }
  function officialLinks() {
    var L = LAB.links || {};
    var box = el("div", { class: "official" });
    box.appendChild(el("span", { class: "official-label" }, tr("Official lab presence")));
    var row = el("span", { class: "official-row" });
    if (L.lab_site) row.appendChild(el("a", { href: L.lab_site, target: "_blank", rel: "noopener" }, tr("Lab website")));
    if (L.pi_homepage) row.appendChild(el("a", { href: L.pi_homepage, target: "_blank", rel: "noopener" }, tr("Prof. Dongrui Wu")));
    if (L.scholar) row.appendChild(el("a", { href: L.scholar, target: "_blank", rel: "noopener" }, tr("Google Scholar")));
    box.appendChild(row);
    return box;
  }
  function pillarTable() {
    var max = topicList.length ? topicList[0][1] : 1;
    var codeByTopic = {};
    PUBS.forEach(function (p) { if (p.code_url) codeByTopic[p.topic] = (codeByTopic[p.topic] || 0) + 1; });
    var table = el("table", { class: "pillars" });
    var head = el("tr", {});
    ["Research area", "", "Papers", "With code"].forEach(function (h, i) { head.appendChild(el("th", { class: i > 1 ? "num" : "" }, tr(h))); });
    table.appendChild(head);
    topicList.forEach(function (row) {
      var topic = row[0], n = row[1];
      var rowTr = el("tr", {});
      var a = el("a", { href: "#papers", onclick: function () { setTopicFilter(topic); } }, tr(topic));
      rowTr.appendChild(el("td", {}, a));
      var bar = el("div", { class: "bar" });
      bar.appendChild(el("span", { style: "width:" + Math.round(100 * n / max) + "%" }));
      rowTr.appendChild(el("td", {}, bar));
      rowTr.appendChild(el("td", { class: "num" }, String(n)));
      rowTr.appendChild(el("td", { class: "num" }, String(codeByTopic[topic] || 0)));
      table.appendChild(rowTr);
    });
    return table;
  }
  // Every approach evaluated in the benchmark, grouped by leaderboard table, for
  // the Overview. The lab's own methods (Prof. Wu's group) are highlighted; the
  // external baselines they are compared against are shown muted alongside.
  function benchApproaches() {
    var wrap = el("div", { class: "lab-methods" }), any = false;
    (BENCH.tables || []).forEach(function (t) {
      // all rows for this table, deduped by method name. A method can appear in
      // several sub-categories — the privacy family, for instance, is measured on
      // three datasets — but it should still show as a single chip. The no-op
      // baseline placeholder ("none", i.e. no alignment / no augmentation) is not
      // an approach, so it is skipped.
      var order = [], byName = {};
      (t.groups || []).forEach(function (g) {
        (g.rows || []).forEach(function (r) {
          if (!r.name || r.name.toLowerCase() === "none") return;
          // remember the method's implementation path so the chip can link to its
          // exact code file; the same method may recur across sub-categories.
          if (byName[r.name]) { byName[r.name].count++; if (!byName[r.name].code && r.code) byName[r.name].code = r.code; }
          else { byName[r.name] = { name: r.name, lab: !!r.lab, count: 1, code: r.code || null }; order.push(r.name); }
        });
      });
      if (!order.length) return;
      any = true;
      // lab-proposed approaches first, then the external baselines, each keeping
      // first-appearance order, so the highlighted ones lead every table.
      var labFirst = order.filter(function (n) { return byName[n].lab; })
        .concat(order.filter(function (n) { return !byName[n].lab; }));
      var box = el("div", { class: "lab-cat" });
      box.appendChild(el("div", { class: "lab-cat-label" }, tr(t.title)));
      var chips = el("div", { class: "lab-chips" });
      labFirst.forEach(function (nm) {
        var m = byName[nm];
        // link each approach name to its exact implementation file (its GitHub
        // code page) rather than to the Benchmark tab.
        var cls = "lab-chip " + (m.lab ? "is-lab" : "is-ext");
        var chip = m.code
          ? el("a", { class: cls, href: REPO_URL + "/blob/main/" + m.code,
              target: "_blank", rel: "noopener", title: tr("Open ") + m.code })
          : el("a", { class: cls, href: "#benchmark" });
        chip.appendChild(el("span", { class: "lm-name" }, m.name));
        chips.appendChild(chip);
      });
      box.appendChild(chips);
      wrap.appendChild(box);
    });
    return any ? wrap : null;
  }
  function overview() {
    var o = document.getElementById("overview");
    o.textContent = "";
    var hero = el("div", { class: "hero" });
    if (LAB.pi_photo) {
      var pic = el("figure", { class: "pi-photo" });
      pic.appendChild(el("img", { src: LAB.pi_photo, alt: LAB.pi || "Principal Investigator", loading: "lazy" }));
      pic.appendChild(el("figcaption", {}, tr(LAB.pi || "")));
      hero.appendChild(pic);
    }
    hero.appendChild(el("h1", {}, tr(LAB.full_name || "HUST BCIML")));
    if (LAB.tagline) hero.appendChild(el("p", { class: "tagline" }, tr(LAB.tagline)));
    var who = el("p", { class: "who" });
    who.appendChild(el("strong", {}, tr(LAB.pi || "")));
    who.appendChild(txt(" · " + tr(LAB.institution || "")));
    hero.appendChild(who);
    if (LAB.repo_intro) hero.appendChild(el("p", { class: "repo-intro" }, tr(LAB.repo_intro)));
    hero.appendChild(officialLinks());
    var stats = el("div", { class: "stats" });
    var labKeys = new Set();
    (BENCH.tables || []).forEach(function (t) { (t.groups || []).forEach(function (g) { (g.rows || []).forEach(function (r) { if (r.lab && r.key) labKeys.add(r.key); }); }); });
    stat(stats, labKeys.size, tr("lab approaches"));
    stat(stats, SITE.n_methods, tr("approaches benchmarked"));
    stat(stats, SITE.n_code, tr("papers with code"));
    stat(stats, SITE.n_papers || PUBS.length, tr("papers indexed"));
    stat(stats, topicList.length, tr("research areas"));
    stat(stats, paradigmList.length, tr("BCI paradigms"));
    hero.appendChild(stats);
    o.appendChild(hero);

    // lead with the full set of benchmarked approaches, grouped by table, with the
    // lab's own methods highlighted and the external baselines muted.
    var lm = benchApproaches();
    if (lm) {
      o.appendChild(el("div", { class: "section-title" }, tr("Approaches in the benchmark")));
      o.appendChild(el("p", { class: "area-note" },
        tr("Every approach evaluated in the benchmark, grouped by pipeline stage. The lab's own methods (Prof. Wu's group) are highlighted; the external baselines they are compared against are shown alongside.")));
      var legend = el("div", { class: "approach-legend" });
      legend.appendChild(el("span", { class: "lgd lgd-lab" }, tr("lab-proposed")));
      legend.appendChild(el("span", { class: "lgd lgd-ext" }, tr("external baseline")));
      o.appendChild(legend);
      o.appendChild(lm);
    }

    // code first: the anchor project + its benchmark
    if (LAB.anchor) {
      o.appendChild(el("div", { class: "section-title" }, tr("Anchor project")));
      var a = LAB.anchor;
      var card = el("div", { class: "card anchor" });
      card.appendChild(el("h3", {}, a.name + (a.stars ? "  ·  " + a.stars + " " + tr("stars") : "")));
      card.appendChild(el("div", { class: "blurb" }, tr(a.blurb || "")));
      var act = el("div", { class: "actions" });
      act.appendChild(el("a", { class: "btn", href: "#benchmark" }, tr("View the benchmark")));
      act.appendChild(el("a", { class: "btn ghost", href: REPO_URL + "/tree/main/" + PKG, target: "_blank", rel: "noopener" }, tr("Benchmark code")));
      card.appendChild(act);
      o.appendChild(card);
    }

    // code: featured repositories, promoted above the paper-count table
    if (LAB.flagships && LAB.flagships.length) {
      o.appendChild(el("div", { class: "section-title" }, tr("Featured code repositories")));
      var g = el("div", { class: "grid" });
      LAB.flagships.forEach(function (f) {
        var c = el("div", { class: "flag" });
        c.appendChild(el("div", { class: "pillar" }, tr(f.pillar || "")));
        c.appendChild(el("h4", {}, el("a", { href: f.url, target: "_blank", rel: "noopener" }, f.name)));
        if (f.blurb) c.appendChild(el("div", { class: "blurb" }, f.blurb));   // per-repo blurb: kept in English
        var badge = f.stars ? (f.stars + " " + tr("stars")) : (f.cites ? (f.cites + " " + tr("citations")) : "");
        if (badge) c.appendChild(el("div", { class: "badge" }, badge));
        g.appendChild(c);
      });
      o.appendChild(g);
    }

    // papers as a paper-to-code map, secondary to the code above
    o.appendChild(el("div", { class: "section-title" }, tr("Browse the lab's work by area")));
    o.appendChild(el("p", { class: "area-note" },
      tr("Publications grouped by research area, with how many have released code. " +
      "Open Papers & Code to search and filter; the official sites above hold the full publication list.")));
    o.appendChild(pillarTable());
  }

  // ---------------- papers & code gallery ----------------
  var state = { q: "", topics: new Set(), paradigms: new Set(), codeOnly: true };
  var listEl, countEl, searchInput, codeCheck;

  function makeChip(kind, label, n) {
    // data-label stays the English value (it is the filter key matched against
    // p.topic / p.paradigm); only the visible text is translated.
    var c = el("span", { class: "chip", "data-kind": kind, "data-label": label,
      onclick: function () { toggleFacet(kind, label); } });
    c.appendChild(txt(tr(label)));
    c.appendChild(el("span", { class: "c" }, String(n)));
    return c;
  }
  function toggleFacet(kind, label) {
    var set = state[kind];
    if (set.has(label)) set.delete(label); else set.add(label);
    syncChips(); applyFilters();
  }
  function syncChips() {
    document.querySelectorAll(".chip").forEach(function (c) {
      c.classList.toggle("on", state[c.dataset.kind].has(c.dataset.label));
    });
  }
  function setTopicFilter(topic) {
    state.q = ""; state.paradigms.clear(); state.codeOnly = false;
    state.topics = new Set([topic]);
    if (searchInput) searchInput.value = "";
    if (codeCheck) codeCheck.checked = false;
    syncChips(); applyFilters();
  }
  function clearAll() {
    state.q = ""; state.topics.clear(); state.paradigms.clear(); state.codeOnly = false;
    if (searchInput) searchInput.value = "";
    if (codeCheck) codeCheck.checked = false;
    syncChips(); applyFilters();
  }
  function applyFilters() {
    var q = state.q;
    var res = PUBS.filter(function (p) {
      if (state.codeOnly && !p.code_url) return false;
      if (state.topics.size && !state.topics.has(p.topic)) return false;
      if (state.paradigms.size) {
        var par = p.paradigm || [], ok = false;
        state.paradigms.forEach(function (x) { if (par.indexOf(x) >= 0) ok = true; });
        if (!ok) return false;
      }
      if (q) {
        var hay = ((p.title || "") + " " + (p.authors || "") + " " + (p.venue || "") + " " + (p.tldr || "")).toLowerCase();
        if (hay.indexOf(q) < 0) return false;
      }
      return true;
    });
    countEl.textContent = tr("showing") + " " + res.length + " " + tr("of") + " " + PUBS.length;
    renderList(res);
  }
  function paperCard(p) {
    var art = el("article", { class: "paper" });
    var head = el("div", { class: "paper-head" });
    if (p.year) head.appendChild(el("span", { class: "year" }, String(p.year)));
    var title = el("h3", { class: "paper-title" });
    if (p.doi) title.appendChild(el("a", { href: "https://doi.org/" + p.doi, target: "_blank", rel: "noopener" }, p.title || ""));
    else title.appendChild(txt(p.title || ""));
    head.appendChild(title);
    art.appendChild(head);

    var meta = [p.authors, p.venue].filter(Boolean).join(" · ");
    if (meta) art.appendChild(el("div", { class: "paper-meta" }, meta));

    var tags = el("div", { class: "paper-tags" });
    // topic / paradigm display text is translated; the value passed to the filter
    // handlers stays the English key. Paper title/authors/venue/tldr stay English.
    if (p.topic) tags.appendChild(el("span", { class: "topic-badge", onclick: function () { setTopicFilter(p.topic); } }, tr(p.topic)));
    (p.paradigm || []).forEach(function (pd) {
      var t = el("span", { class: "tag", onclick: function () { toggleFacet("paradigms", pd); } }, tr(pd));
      t.style.background = "hsl(" + hue(pd) + " 65% 90%)";
      t.style.color = "hsl(" + hue(pd) + " 55% 30%)";
      tags.appendChild(t);
    });
    // code-first: the code link leads, then the paper
    var links = el("span", { class: "paper-links" });
    if (p.code_url) links.appendChild(el("a", { class: "codelink", href: p.code_url, target: "_blank", rel: "noopener" }, tr("code")));
    if (p.doi) links.appendChild(el("a", { href: "https://doi.org/" + p.doi, target: "_blank", rel: "noopener" }, tr("paper")));
    if (!p.code_url) links.appendChild(el("span", { class: "nocode" }, tr("no code")));
    tags.appendChild(links);
    art.appendChild(tags);

    if (p.tldr) {
      var tldr = el("p", { class: "tldr clamp" }, p.tldr);   // TL;DR kept in English
      var more = el("button", { class: "more" }, tr("more"));
      more.addEventListener("click", function () {
        var clamped = tldr.classList.toggle("clamp");
        more.textContent = clamped ? tr("more") : tr("less");
      });
      art.appendChild(tldr); art.appendChild(more);
    }
    return art;
  }
  function renderList(res) {
    listEl.textContent = "";
    if (!res.length) { listEl.appendChild(el("div", { class: "empty" }, tr("No papers match these filters."))); return; }
    var frag = document.createDocumentFragment();
    res.forEach(function (p) { frag.appendChild(paperCard(p)); });
    listEl.appendChild(frag);
  }
  function papers() {
    var P = document.getElementById("papers"), L = LAB.links || {};
    P.textContent = "";
    P.appendChild(el("div", { class: "section-title" }, tr("Papers & code gallery")));

    var note = el("p", { class: "gallery-note" });
    note.appendChild(txt(tr("The lab's publications, each linked to its released code where available. Showing the ")));
    note.appendChild(el("strong", {}, String(SITE.n_code || 0) + tr(" with public code")));
    note.appendChild(txt(tr("; untick “has code” for all ") + (SITE.n_papers || PUBS.length) +
      tr(". The complete, authoritative publication list is on the ")));
    if (L.lab_site) note.appendChild(el("a", { href: L.lab_site, target: "_blank", rel: "noopener" }, tr("lab website")));
    note.appendChild(txt(tr(" and ")));
    if (L.pi_homepage) note.appendChild(el("a", { href: L.pi_homepage, target: "_blank", rel: "noopener" }, tr("Prof. Wu's homepage")));
    note.appendChild(txt(tr(".")));
    P.appendChild(note);

    var controls = el("div", { class: "controls" });
    var searchrow = el("div", { class: "searchrow" });
    searchInput = el("input", { class: "search", type: "search", placeholder: tr("Search title, authors, venue, summary…") });
    searchInput.addEventListener("input", function () { state.q = searchInput.value.trim().toLowerCase(); applyFilters(); });
    if (state.q) searchInput.value = state.q;   // preserve query text across re-render
    searchrow.appendChild(searchInput);
    var tg = el("label", { class: "toggle" });
    codeCheck = el("input", { type: "checkbox" });
    codeCheck.checked = state.codeOnly;
    codeCheck.addEventListener("change", function () { state.codeOnly = codeCheck.checked; applyFilters(); });
    tg.appendChild(codeCheck); tg.appendChild(txt(tr("has code")));
    searchrow.appendChild(tg);
    searchrow.appendChild(el("button", { class: "clearbtn", onclick: clearAll }, tr("Show all")));
    countEl = el("span", { class: "count" });
    searchrow.appendChild(countEl);
    controls.appendChild(searchrow);

    controls.appendChild(el("div", { class: "chip-legend" }, tr("Research area")));
    var tc = el("div", { class: "chips" });
    topicList.forEach(function (r) { tc.appendChild(makeChip("topics", r[0], r[1])); });
    controls.appendChild(tc);
    controls.appendChild(el("div", { class: "chip-legend" }, tr("BCI paradigm")));
    var pc = el("div", { class: "chips" });
    paradigmList.forEach(function (r) { pc.appendChild(makeChip("paradigms", r[0], r[1])); });
    controls.appendChild(pc);
    P.appendChild(controls);

    listEl = el("div", { class: "papers-list" });
    P.appendChild(listEl);
    syncChips();
    applyFilters();
  }

  // ---------------- benchmark ----------------
  function fmtAcc(mean, std) {
    if (mean == null) return null;
    return std != null ? mean.toFixed(2) + " ± " + std.toFixed(2) : mean.toFixed(2);
  }
  function codeLink(code) {
    if (!code) return null;
    return el("a", { class: "codelink code-impl", href: REPO_URL + "/blob/main/" + code,
      target: "_blank", rel: "noopener", title: tr("Open ") + code }, tr("code"));
  }
  function paperLink(r) {
    if (!r.doi) return null;
    return el("a", { class: "codelink paper-impl", href: "https://doi.org/" + r.doi,
      target: "_blank", rel: "noopener", title: tr("Open the paper") }, tr("paper"));
  }
  // one per-dataset accuracy cell: the acc ± std on top, the coloured Δ vs that
  // dataset's baseline beneath. An absent cell (method inapplicable here) is n/a.
  function accCell(cell, delta, isBaseline) {
    var td = el("td", { class: "acc-cell" });
    var s = cell ? fmtAcc(cell.mean, cell.std) : null;
    if (s == null) { td.classList.add("na"); td.appendChild(el("span", { class: "na-dash" }, tr("n/a"))); return td; }
    td.appendChild(el("div", { class: "cell-acc" }, s));
    if (isBaseline) td.appendChild(el("div", { class: "cell-delta base" }, tr("baseline")));
    else if (delta != null) {
      var cls = delta > 0 ? "pos" : (delta < 0 ? "neg" : "zero");
      td.appendChild(el("div", { class: "cell-delta " + cls }, (delta > 0 ? "+" : "") + delta.toFixed(2)));
    }
    return td;
  }
  // one method's row: name with inline description, citation, and links to its
  // implementation and paper; then one accuracy/Δ cell per dataset.
  function methodRowMulti(r, datasets) {
    // method name / desc / ref are kept in English; only the "lab" badge translates.
    var rowTr = el("tr", { class: "lb-row" + (r.lab ? " lab" : "") });
    var name = el("td", {});
    var line1 = el("div", { class: "m-line" });
    line1.appendChild(el("span", { class: "m-name" }, r.name));
    if (r.lab) line1.appendChild(el("span", { class: "lab-badge" }, tr("lab")));
    var cl = codeLink(r.code);
    if (cl) line1.appendChild(cl);
    var pl = paperLink(r);
    if (pl) line1.appendChild(pl);
    name.appendChild(line1);
    if (r.desc) name.appendChild(el("div", { class: "m-desc" }, r.desc));
    if (r.ref) name.appendChild(el("div", { class: "m-ref" }, r.ref));
    rowTr.appendChild(name);
    datasets.forEach(function (d) {
      var cell = r.acc ? r.acc[d] : null;
      var delta = r.delta ? r.delta[d] : null;
      rowTr.appendChild(accCell(cell, delta, r.isBaseline && cell != null));
    });
    return rowTr;
  }
  // a full-width reference line inside the table (a group's baseline context),
  // listing that baseline's accuracy on each dataset.
  function refTrMulti(ref, datasets, label, ncol) {
    var parts = datasets.map(function (d) {
      var c = ref.acc ? ref.acc[d] : null;
      return d + " " + (c ? fmtAcc(c.mean, c.std) : tr("n/a"));
    });
    var rowTr = el("tr", { class: "ref-tr" });
    // label ("baseline"/"reference") translates; ref.name embeds method names, kept English.
    rowTr.appendChild(el("td", { colspan: String(ncol) },
      tr(label || "reference") + ": " + ref.name + " — " + parts.join("  ·  ")));
    return rowTr;
  }
  // one leaderboard table rendered as a SINGLE <table> with a Method column plus
  // one column per dataset. Named sub-categories (the Transfer families) appear as
  // section-header rows inside that one table, each measured against its own
  // per-dataset baseline.
  function renderTableMulti(t, datasets, host) {
    var groups = t.groups || [];
    var grouped = groups.length > 1 || (groups[0] && groups[0].subcat);
    var ncol = 1 + datasets.length;
    var table = el("table", { class: "lb multi" + (grouped ? " grouped" : "") });
    var head = el("tr", {});
    head.appendChild(el("th", {}, tr("Approach")));
    datasets.forEach(function (d) { head.appendChild(el("th", { class: "ds-col" }, d)); });   // dataset names: identifiers, kept as-is
    table.appendChild(head);
    groups.forEach(function (g) {
      if (g.subcat) {
        var hr = el("tr", { class: "subcat-tr" });
        var cell = el("td", { colspan: String(ncol) });
        cell.appendChild(el("span", { class: "subcat-name" }, tr(g.subcat)));
        if (g.blurb) cell.appendChild(el("span", { class: "subcat-blurb" }, tr(g.blurb)));
        hr.appendChild(cell);
        table.appendChild(hr);
      }
      g.rows.forEach(function (r) { table.appendChild(methodRowMulti(r, datasets)); });
      if (g.reference) table.appendChild(refTrMulti(g.reference, datasets, "baseline", ncol));
    });
    host.appendChild(table);
  }
  // compact overview of the datasets the benchmark spans, rendered at the top of
  // the Benchmark tab so the three-dataset coverage is explicit (the pipeline-stage
  // tables run on the primary dataset; the privacy-preserving family runs on all three).
  function renderDatasets(list, host) {
    if (!list || !list.length) return;
    host.appendChild(el("div", { class: "section-title" }, tr("Datasets")));
    // dynamic intro: the number is dropped in between two translated fragments.
    host.appendChild(el("p", { class: "bench-intro" },
      tr("The benchmark spans ") + list.length +
      tr(" MOABB motor-imagery EEG datasets, all evaluated cross-subject (leave-one-subject-out). Accuracies are comparable only within the same dataset and class count.")));
    var table = el("table", { class: "pillars datasets" });
    var head = el("tr", {});
    ["Dataset", "Subjects", "Channels", "Rate", "Classes", "Chance", "Trials/subj"]
      .forEach(function (h, i) { head.appendChild(el("th", { class: i ? "num" : "" }, tr(h))); });
    table.appendChild(head);
    list.forEach(function (d) {
      var rowTr = el("tr", {});
      var name = el("td", {});
      name.appendChild(el("div", { class: "ds-name" }, d.name));   // dataset name: identifier, kept as-is
      if (d.role) name.appendChild(el("div", { class: "ds-role" }, tr(d.role)));
      rowTr.appendChild(name);
      [d.subjects, d.channels, (d.sfreq != null ? d.sfreq + " Hz" : null),
       d.classes, d.chance, d.trials].forEach(function (v) {
        rowTr.appendChild(el("td", { class: "num" }, v == null ? "—" : String(v)));
      });
      table.appendChild(rowTr);
    });
    host.appendChild(table);
  }
  // per-dataset frame for the ensemble-learning category: class count / chance,
  // single-source mean, Centralized Training, and the majority-voting baseline. Shown
  // above that table inside the Benchmark view (it is a category, not a separate tab).
  function renderEnsembleContext(t, host) {
    if (!t.context) return;
    var ctxWrap = el("div", { class: "ens-context" });
    DS.forEach(function (d) {
      var c = t.context[d]; if (!c) return;
      var card = el("div", { class: "ens-ctx-card" });
      card.appendChild(el("div", { class: "ens-ctx-ds" }, d));   // dataset name: identifier
      card.appendChild(el("div", { class: "ens-ctx-meta" },
        (c.classes || "") + (c.chance != null ? " · " + tr("chance") + " " + c.chance + "%" : "")));
      var nums = el("div", { class: "ens-ctx-nums" });
      function num(lbl, v) {
        var s = el("div", { class: "ens-ctx-num" });
        s.appendChild(el("span", { class: "l" }, tr(lbl)));
        s.appendChild(el("span", { class: "v" }, v == null ? "—" : v.toFixed(2)));
        return s;
      }
      nums.appendChild(num("single-source", c.single_source));
      nums.appendChild(num("Centralized Training", c.centralized));
      nums.appendChild(num("majority voting", c.voting));
      card.appendChild(nums);
      ctxWrap.appendChild(card);
    });
    host.appendChild(ctxWrap);
  }
  function benchmark() {
    var B = document.getElementById("benchmark");
    B.textContent = "";
    var m = BENCH.meta || {}, lib = BENCH.library || {};

    // what the benchmark is (the code), stated without a separate product name
    if (lib.title) {
      B.appendChild(el("div", { class: "section-title" }, tr("The benchmark")));
      var li = el("div", { class: "card lib-intro" });
      li.appendChild(el("h3", {}, tr(lib.title)));
      if (lib.tagline) li.appendChild(el("div", { class: "blurb" }, tr(lib.tagline)));
      if (lib.pipeline && lib.pipeline.length) {
        var st = el("div", { class: "stages" });
        lib.pipeline.forEach(function (s, i) {
          if (i) st.appendChild(el("span", { class: "stage-arrow" }, "→"));
          st.appendChild(el("span", { class: "stage" }, tr(s)));
        });
        if (lib.driver) { st.appendChild(el("span", { class: "stage-driver" }, tr("trained under"))); st.appendChild(el("span", { class: "stage" }, tr(lib.driver))); }
        li.appendChild(st);
      }
      var la = el("div", { class: "actions" });
      la.appendChild(el("a", { class: "btn", href: REPO_URL + "/tree/main/" + PKG, target: "_blank", rel: "noopener" }, tr("Benchmark code")));
      la.appendChild(el("a", { class: "btn ghost", href: REPO_URL + "/blob/main/" + PKG + "/README.md", target: "_blank", rel: "noopener" }, tr("README")));
      la.appendChild(el("a", { class: "btn ghost", href: REPO_URL + "/blob/main/" + PKG + "/RESULTS.md", target: "_blank", rel: "noopener" }, tr("RESULTS.md")));
      la.appendChild(el("a", { class: "btn ghost", href: REPO_URL + "/blob/main/references.bib", target: "_blank", rel: "noopener" }, tr("References (BibTeX)")));
      li.appendChild(la);
      B.appendChild(li);
    }

    renderDatasets(m.datasets, B);

    B.appendChild(el("div", { class: "section-title" }, tr("Controlled-comparison leaderboard")));
    var guide = el("details", { class: "bench-guide" });
    guide.appendChild(el("summary", {}, tr("How to read this leaderboard")));
    guide.appendChild(el("p", {},
      tr("Each table varies one stage of the pipeline and holds the rest at the default — Euclidean-aligned " +
      "trials, an EEGNet backbone, standard supervised training — so every row differs from its baseline in " +
      "exactly one way. The three columns are the three datasets; beneath each accuracy, Δ is the change " +
      "versus that dataset's baseline. Every table is two-class (chance 50%) on all three datasets — the " +
      "pipeline-stage tables, the source-only, unsupervised-adaptation, source-free and test-time transfer " +
      "families, the privacy-preserving family, and the ensemble-learning table — so the columns are directly " +
      "comparable throughout. Each family is measured against its own baseline: the transfer families against " +
      "ERM, the privacy-preserving family against Centralized Training, and the ensemble table against " +
      "majority voting. Every row links to its implementation and its paper.")));
    B.appendChild(guide);

    (BENCH.tables || []).forEach(function (t) {
      var box = el("div", { class: "bench-table" });
      box.appendChild(el("h3", {}, tr(t.title)));
      if (t.blurb) box.appendChild(el("div", { class: "blurb" }, tr(t.blurb)));
      if (t.id === "ensemble") renderEnsembleContext(t, box);   // per-dataset context cards
      renderTableMulti(t, DS, box);
      B.appendChild(box);
    });
  }

  // (the ensemble-learning table renders inline in the Benchmark view as a
  //  category, via renderEnsembleContext + renderTableMulti in the benchmark() builder.)

  // ---------------- render + language toggle ----------------
  // Rebuild all three views + header/footer from scratch. Safe to call repeatedly;
  // each builder clears its own container first. Filter state in `state` persists
  // across calls, so switching language keeps the papers filters intact.
  function render() {
    header();
    footer();
    overview();
    papers();
    benchmark();
    applyTabLabels();
  }

  // The three tab labels live as static text in index.html; translate them here so
  // they follow the language toggle without moving them out of the markup.
  function applyTabLabels() {
    document.querySelectorAll("#tabs .tab").forEach(function (b) {
      var v = b.dataset.view;
      b.textContent = v === "overview" ? tr("Overview")
        : v === "benchmark" ? tr("Benchmark")
        : v === "papers" ? tr("Papers & Code")
        : b.textContent;
    });
  }

  function setLang(next) {
    LANG = next;
    try { localStorage.setItem("lang", LANG); } catch (e) {}
    document.documentElement.lang = (LANG === "zh") ? "zh-Hans" : "en";
    updateLangToggle();
    var current = location.hash.slice(1) || "overview";
    render();
    activate(current);   // re-apply the active tab after the rebuild
  }

  var langToggle = null;
  function updateLangToggle() {
    if (!langToggle) return;
    // show the language you would switch TO
    langToggle.textContent = (LANG === "zh") ? "EN" : "中文";
    langToggle.setAttribute("aria-label", (LANG === "zh") ? "Switch to English" : "切换到中文");
    langToggle.setAttribute("title", (LANG === "zh") ? "Switch to English" : "切换到中文");
  }
  (function mountLangToggle() {
    // inject a toggle button styled like the tabs, into the header nav
    var tabs = document.getElementById("tabs");
    langToggle = el("button", { class: "tab lang-toggle", type: "button",
      onclick: function () { setLang(LANG === "zh" ? "en" : "zh"); } });
    if (tabs) tabs.appendChild(langToggle);
    updateLangToggle();
  })();

  // ---------------- init ----------------
  document.documentElement.lang = (LANG === "zh") ? "zh-Hans" : "en";
  render();
  activate(location.hash.slice(1) || "overview");
})();

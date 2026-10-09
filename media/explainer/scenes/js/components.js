// Reusable, time-driven building blocks of the TabAgent explainer. Each component builds its DOM once and
// exposes update(...) that sets its state for one frame.
(function (g) {
  "use strict";
  const { h, place, tf, vis, svgAttr, P, E, ENV, lerp, clamp, words, revealWords, pulse } = g.K;

  const C = {
    ink: "#0B1020", ink2: "#141A2E", paper: "#F7F8FC", text2: "#CBD5E1", dim: "#94A3B8", slate: "#64748B",
    schema: "#22D3EE", synth: "#A78BFA", head: "#F59E0B", llm: "#F43F5E", ok: "#34D399",
    line: "rgba(230,233,242,0.09)", line2: "rgba(230,233,242,0.16)",
  };
  const rgba = (hex, a) => {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  };

  // ---------------------------------------------------------------- icons (24 px grid, 1.5 px strokes)
  const ICON = {
    check: "M5 12.5l4.2 4.2L19 7.5",
    clock: "M12 3.5a8.5 8.5 0 1 0 0 17a8.5 8.5 0 1 0 0-17M12 7.5V12l3 2",
    coin: "M12 3.5a8.5 8.5 0 1 0 0 17a8.5 8.5 0 1 0 0-17M14.6 9.2c-.5-.9-1.4-1.4-2.6-1.4-1.5 0-2.5.8-2.5 1.9 0 2.6 5.1 1.4 5.1 4.2 0 1.1-1.1 2-2.6 2-1.3 0-2.3-.6-2.8-1.6M12 6.3v1.5m0 8.4v1.5",
    lock: "M7 11h10a1.5 1.5 0 0 1 1.5 1.5v6A1.5 1.5 0 0 1 17 20H7a1.5 1.5 0 0 1-1.5-1.5v-6A1.5 1.5 0 0 1 7 11zM8.5 11V8.5a3.5 3.5 0 0 1 7 0V11",
    scan: "M4 8.5v-3A1.5 1.5 0 0 1 5.5 4h3M15.5 4h3A1.5 1.5 0 0 1 20 5.5v3M20 15.5v3a1.5 1.5 0 0 1-1.5 1.5h-3M8.5 20h-3A1.5 1.5 0 0 1 4 18.5v-3M7.5 12h9",
    code: "M8.5 8l-4 4 4 4M15.5 8l4 4-4 4M13.5 5.5l-3 13",
    shield: "M12 3.5l7 2.6v5.4c0 4.2-2.9 7.6-7 9-4.1-1.4-7-4.8-7-9V6.1zM8.8 12.2l2.2 2.2 4.3-4.4",
    scale: "M12 4.5v15M8 19.5h8M5 8h14M12 4.5L5 8M12 4.5L19 8M5 8l-2.6 6.2a2.8 2.8 0 0 0 5.2 0zM19 8l-2.6 6.2a2.8 2.8 0 0 0 5.2 0z",
    stopwatch: "M12 6a7.5 7.5 0 1 0 0 15a7.5 7.5 0 1 0 0-15M12 13.5V10M10 3h4M18.2 6.8l1.3-1.3",
    cpu: "M8.5 7h7A1.5 1.5 0 0 1 17 8.5v7a1.5 1.5 0 0 1-1.5 1.5h-7A1.5 1.5 0 0 1 7 15.5v-7A1.5 1.5 0 0 1 8.5 7zM10 4v3M14 4v3M10 17v3M14 17v3M4 10h3M4 14h3M17 10h3M17 14h3",
    copy: "M10 8h8.5A1.5 1.5 0 0 1 20 9.5V18a1.5 1.5 0 0 1-1.5 1.5H10A1.5 1.5 0 0 1 8.5 18V9.5A1.5 1.5 0 0 1 10 8zM5 15.5V6A1.5 1.5 0 0 1 6.5 4.5H15",
    table: "M5.5 5h13A1.5 1.5 0 0 1 20 6.5v11a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 17.5v-11A1.5 1.5 0 0 1 5.5 5zM4 10h16M4 14.5h16M10 5v14",
    arrow: "M5 12h14M13.5 6.5L19 12l-5.5 5.5",
    arrowUR: "M7 17L17 7M9 7h8v8",
    x: "M7 7l10 10M17 7L7 17",
    tree: "M12 7.5v3M12 10.5H7v4M12 10.5h5v4M12 3.5a2 2 0 1 1 0 4a2 2 0 1 1 0-4M7 14.5a2 2 0 1 1 0 4a2 2 0 1 1 0-4M17 14.5a2 2 0 1 1 0 4a2 2 0 1 1 0-4",
    gate: "M4 12h6M10 12l4-5h6M10 12l4 5h6",
  };
  function icon(parent, name, size, color, sw = 1.5) {
    const s = h("svg", { attrs: { viewBox: "0 0 24 24", width: size, height: size, fill: "none" } }, parent);
    s.style.flex = "none";
    const p = h("path", {
      attrs: { d: ICON[name], stroke: color, "stroke-width": sw, "stroke-linecap": "round", "stroke-linejoin": "round" },
    }, s);
    s.path = p;
    return s;
  }

  // ---------------------------------------------------------------- brand: mark + wordmark
  // The mark: a 3x3 table whose last column is the amber decision (brand spec; same geometry as the
  // repository's tabagent-mark.svg).
  const MARK_CELLS = [
    [12, 15, 11, "#F7F8FC", 0.92], [26, 15, 12, "#F7F8FC", 0.42], [41, 15, 11, "#F59E0B", 0.38],
    [12, 28.5, 11, "#F7F8FC", 0.92], [26, 28.5, 12, "#F7F8FC", 0.42], [41, 28.5, 11, "#F59E0B", 1],
    [12, 42, 11, "#F7F8FC", 0.92], [26, 42, 12, "#F7F8FC", 0.42], [41, 42, 11, "#F59E0B", 0.38],
  ];
  function Mark(parent, size) {
    const s = h("svg", { attrs: { viewBox: "0 0 64 64", width: size, height: size } }, parent);
    h("rect", { attrs: { width: 64, height: 64, rx: 15, fill: C.ink2 } }, s);
    const border = h("rect", {
      attrs: { x: 0.75, y: 0.75, width: 62.5, height: 62.5, rx: 14.25, fill: "none", stroke: "#2A3350", "stroke-width": 1.5 },
    }, s);
    const cells = MARK_CELLS.map(([x, y, w, c, a]) =>
      h("rect", { attrs: { x, y, width: w, height: 7.5, rx: 2.6, fill: c, "fill-opacity": a } }, s));
    return {
      el: s, cells, border,
      // k: 0..1 assembly; cells pop in by column, the amber decision last
      assemble(k) {
        cells.forEach((c, i) => {
          const col = i % 3, row = Math.floor(i / 3);
          const kk = clamp((k - (col * 0.22 + row * 0.06)) / 0.34);
          const e = E.brand(kk);
          c.style.opacity = e.toFixed(3);
          c.style.transformOrigin = `${MARK_CELLS[i][0] + MARK_CELLS[i][2] / 2}px ${MARK_CELLS[i][1] + 3.75}px`;
          c.style.transform = `scale(${(0.6 + 0.4 * e).toFixed(3)})`;
        });
      },
    };
  }
  function Wordmark(parent, size, opt = {}) {
    const wrap = h("div", { style: { display: "flex", alignItems: "center", gap: size * 0.32 + "px" } }, parent);
    const mark = Mark(wrap, size * 1.18);
    const txt = h("div", {
      style: {
        fontWeight: 800, fontSize: size + "px", letterSpacing: "-0.025em", lineHeight: 1, whiteSpace: "nowrap",
        paddingBottom: size * 0.04 + "px",
      },
    }, wrap);
    const tab = h("span", { text: "Tab", style: { color: C.paper } }, txt);
    const agent = h("span", {
      text: "Agent",
      style: {
        background: `linear-gradient(90deg, ${C.schema}, ${C.synth} 52%, ${C.head})`,
        WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent",
      },
    }, txt);
    return { el: wrap, mark, txt, tab, agent };
  }

  // ---------------------------------------------------------------- background
  function Background(stage, W, H) {
    const root = h("div", { cls: "abs", style: { width: W + "px", height: H + "px" } }, stage);
    const glows = {};
    const glowDefs = { llm: C.llm, schema: C.schema, synth: C.synth, head: C.head, ok: C.ok, paper: "#8FA3D9" };
    for (const k in glowDefs) {
      const d = h("div", { cls: "glow" }, root);
      d.style.background = `radial-gradient(closest-side, ${rgba(glowDefs[k], 0.22)}, ${rgba(glowDefs[k], 0.07)} 55%, transparent)`;
      glows[k] = d;
    }
    const dots = h("div", { cls: "bg-dots" }, root);
    h("div", { cls: "vignette" }, root);
    return {
      root, glows, dots,
      // set(name, x, y, w, hgt, opacity)
      glow(name, x, y, w, hgt, o) {
        const d = glows[name];
        place(d, x - w / 2, y - hgt / 2, w, hgt);
        vis(d, o);
      },
    };
  }

  // ---------------------------------------------------------------- top bar: brand + phase rail
  const PHASES = [
    { key: "schema", name: "TabSchema", color: C.schema, sub: "discovers features" },
    { key: "synth", name: "TabSynth", color: C.synth, sub: "augments the rows" },
    { key: "head", name: "TabHead", color: C.head, sub: "decides in milliseconds" },
  ];
  function Rail(parent, rightX, y) {
    const wrap = h("div", { cls: "abs", style: { display: "flex", alignItems: "center", gap: "10px" } }, parent);
    const items = PHASES.map((ph, i) => {
      if (i) {
        const c = h("div", { style: { width: "26px", height: "1.5px", background: C.line2, borderRadius: "1px" } }, wrap);
        c.dataset.conn = "1";
      }
      const pill = h("div", {
        style: {
          display: "flex", alignItems: "center", gap: "9px", height: "38px", padding: "0 15px 0 8px",
          borderRadius: "19px", border: `1px solid ${C.line2}`, background: "rgba(255,255,255,0.02)",
          fontWeight: 600, fontSize: "16px", color: C.dim, whiteSpace: "nowrap",
        },
      }, wrap);
      const b = h("div", {
        cls: "badge", text: String(i + 1),
        style: { width: "24px", height: "24px", fontSize: "13px", background: "rgba(148,163,184,0.25)", color: C.ink },
      }, pill);
      const name = h("span", { text: ph.name }, pill);
      return { pill, b, name, ph };
    });
    const conns = [...wrap.querySelectorAll("[data-conn]")];
    return {
      el: wrap,
      update(t, o, active /* float: 0,1,2; -1 none */, y0 = y) {
        const w = wrap.offsetWidth;
        place(wrap, rightX - w, y0);
        vis(wrap, o);
        items.forEach((it, i) => {
          const a = clamp(1 - Math.abs(active - i)); // 1 when this phase is active
          const done = active >= i + 0.5 ? 1 : 0;
          const col = it.ph.color;
          it.pill.style.borderColor = a > 0.01 ? rgba(col, 0.25 + 0.55 * a) : done ? rgba(col, 0.28) : C.line2;
          it.pill.style.background = a > 0.01 ? rgba(col, 0.1 * a) : "rgba(255,255,255,0.02)";
          it.name.style.color = a > 0.5 ? col : done ? C.text2 : C.dim;
          it.b.style.background = a > 0.01 || done ? rgba(col, 0.35 + 0.65 * Math.max(a, done * 0.6)) : "rgba(148,163,184,0.28)";
        });
        conns.forEach((c, i) => {
          c.style.background = active >= i + 0.5 ? rgba(PHASES[i].color, 0.55) : C.line2;
        });
      },
    };
  }

  // ---------------------------------------------------------------- narration column (left)
  function Narration(parent, cues, x, y, w) {
    const items = cues.map((c) => {
      const el = h("div", { cls: "abs headline" }, parent);
      place(el, x, y, w);
      // "\n" in a cue forces a line break on screen (captions join the lines)
      let ws = [];
      for (const ln of c.text.split("\n")) ws = ws.concat(words(h("div", {}, el), ln));
      vis(el, 0);
      return { c, el, ws };
    });
    return {
      items,
      update(t) {
        for (const it of items) {
          const { start, end } = it.c;
          if (t < start - 0.01 || t > end + 0.01) { vis(it.el, 0); continue; }
          const out = P(t, end - 0.32, 0.32, E.inOut);
          vis(it.el, 1 - out);
          it.el.style.transform = `translateY(${(-14 * out).toFixed(2)}px)`;
          revealWords(it.ws, t, start, { stagger: 0.026, dur: 0.5, dy: 20 });
        }
      },
    };
  }

  // eyebrow for a phase: badge + name + sub
  function PhaseEyebrow(parent, idx, x, y) {
    const ph = PHASES[idx];
    const wrap = h("div", { cls: "abs", style: { display: "flex", alignItems: "center", gap: "12px" } }, parent);
    place(wrap, x, y);
    h("div", {
      cls: "badge", text: String(idx + 1),
      style: { width: "30px", height: "30px", fontSize: "16px", background: ph.color },
    }, wrap);
    h("div", { text: ph.name, style: { fontWeight: 700, fontSize: "26px", letterSpacing: "-0.01em", color: ph.color } }, wrap);
    h("div", { text: ph.sub, style: { fontWeight: 500, fontSize: "20px", color: C.dim, marginLeft: "2px" } }, wrap);
    return wrap;
  }
  function TextEyebrow(parent, text, x, y, color = C.dim) {
    const e = h("div", { cls: "abs eyebrow", text }, parent);
    e.style.color = color;
    place(e, x, y);
    return e;
  }

  // ---------------------------------------------------------------- the LLM: a rose chat bubble
  function LLMBubble(parent, cx, cy, size) {
    const w = size, hh = size * 0.82;
    const wrap = h("div", { cls: "abs" }, parent);
    place(wrap, cx - w / 2, cy - hh / 2, w, hh);
    const s = h("svg", { attrs: { viewBox: "0 0 220 180", width: w, height: hh } }, wrap);
    const glow = h("path", {
      attrs: {
        d: "M44 10H176A34 34 0 0 1 210 44V106A34 34 0 0 1 176 140H92L54 170L60 140H44A34 34 0 0 1 10 106V44A34 34 0 0 1 44 10Z",
        fill: rgba(C.llm, 0.09), stroke: C.llm, "stroke-width": 2, "stroke-linejoin": "round",
      },
    }, s);
    const dots = [80, 110, 140].map((x) => h("circle", { attrs: { cx: x, cy: 75, r: 8.5, fill: C.llm } }, s));
    return {
      el: wrap, dots, body: glow,
      update(t, o, think, sc = 1) {
        vis(wrap, o);
        wrap.style.transformOrigin = "50% 50%";
        wrap.style.transform = `scale(${sc.toFixed(4)})`;
        // deliberate thinking: one slow wave across the three dots (~1.2 s period); smooth, never a flash
        dots.forEach((d, i) => {
          const k = think ? pulse(t, 1.2, -i * 0.18) : 1;
          d.setAttribute("opacity", (0.3 + 0.7 * k * (think ? 1 : 0) + (think ? 0 : 0.7)).toFixed(3));
          d.setAttribute("cy", (75 - 5 * k * (think ? 1 : 0)).toFixed(2));
        });
      },
    };
  }

  // ---------------------------------------------------------------- the head: a decision-tree glyph
  function TreeGlyph(parent, x, y, w, color = C.head) {
    const hh = w * (200 / 240);
    const wrap = h("div", { cls: "abs" }, parent);
    place(wrap, x, y, w, hh);
    const s = h("svg", { attrs: { viewBox: "0 0 240 200", width: w, height: hh } }, wrap);
    const N = [
      [100, 8], // root
      [40, 86], [160, 86],
      [10, 164], [70, 164], [130, 164], [190, 164],
    ];
    const E2 = [
      [0, 1, "M120 36V61H60V86"], [0, 2, "M120 36V61H180V86"],
      [1, 3, "M60 114V139H30V164"], [1, 4, "M60 114V139H90V164"],
      [2, 5, "M180 114V139H150V164"], [2, 6, "M180 114V139H210V164"],
    ];
    const edges = E2.map(([a, b, d]) => {
      const p = h("path", { attrs: { d, fill: "none", stroke: color, "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round" } }, s);
      p.setAttribute("stroke-dasharray", "200");
      return { a, b, p };
    });
    const nodes = N.map(([nx, ny], i) => {
      const r = h("rect", {
        attrs: {
          x: nx, y: ny, width: 40, height: 28, rx: 8,
          fill: rgba(color, i === 0 ? 0.22 : 0.1), stroke: color, "stroke-width": 1.75,
        },
      }, s);
      r.style.transformOrigin = `${nx + 20}px ${ny + 14}px`;
      return r;
    });
    return {
      el: wrap, nodes, edges,
      // grow: 0..1 builds the tree top-down; path: node indices to light; hl: 0..1
      update(o, grow, path = [], hl = 0, glowK = 0) {
        vis(wrap, o);
        const lvl = (i) => (i === 0 ? 0 : i < 3 ? 1 : 2);
        nodes.forEach((n, i) => {
          const k = E.snap(clamp((grow - lvl(i) * 0.3) / 0.35));
          n.style.transform = `scale(${(0.4 + 0.6 * k).toFixed(3)})`;
          n.style.opacity = k.toFixed(3);
          const on = path.includes(i) ? hl : 0;
          n.setAttribute("fill", rgba(color, (i === 0 ? 0.22 : 0.1) + 0.78 * on));
        });
        edges.forEach((e) => {
          const k = clamp((grow - lvl(e.a) * 0.3 - 0.12) / 0.3);
          e.p.setAttribute("stroke-dashoffset", (200 * (1 - E.out(k))).toFixed(1));
          const on = path.includes(e.a) && path.includes(e.b) ? hl : 0;
          e.p.setAttribute("stroke-width", (2 + 1.5 * on).toFixed(2));
          e.p.setAttribute("stroke-opacity", (0.55 + 0.45 * Math.max(on, glowK)).toFixed(3));
        });
      },
    };
  }

  // ---------------------------------------------------------------- the feature card (a table whose cells
  // are positioned explicitly, so columns can open and rows can be inserted smoothly)
  function Table(parent, spec) {
    const { cols, rows } = spec;
    const rowH = spec.rowH || 50, headH = spec.headH || 44, groupH = spec.groupH || 36;
    const root = h("div", { cls: "abs" }, parent);
    const frame = h("div", { cls: "abs card" }, root);
    const clip = h("div", { cls: "abs tbl" }, root);
    // column highlights (behind cells)
    const colHl = {};
    for (const c of cols) colHl[c.key] = h("div", { cls: "abs", style: { borderRadius: "0px" } }, clip);
    // group header
    const groups = {};
    for (const gdef of spec.groups || []) {
      const gl = h("div", { cls: "abs" }, clip);
      const lab = h("div", {
        text: gdef.label,
        style: {
          position: "absolute", left: "14px", top: "9px", fontWeight: 600, fontSize: "13px", letterSpacing: "0.1em",
          textTransform: "uppercase", color: gdef.color, whiteSpace: "nowrap",
        },
      }, gl);
      const bar = h("div", { style: { position: "absolute", left: "10px", right: "10px", bottom: "0px", height: "2px", borderRadius: "2px", background: gdef.color } }, gl);
      groups[gdef.key] = { gl, lab, bar, def: gdef };
    }
    // header cells
    const head = h("div", { cls: "abs" }, clip);
    const headLine = h("div", { cls: "abs", style: { height: "1px", background: C.line2 } }, clip);
    const hcells = {};
    for (const c of cols) {
      const cell = h("div", { cls: "cell hcell" }, head);
      const a = h("span", { text: c.label }, cell);
      let b = null;
      if (c.alt) {
        b = h("span", { text: c.alt, style: { position: "absolute", left: "14px" } }, cell);
      }
      hcells[c.key] = { cell, a, b };
    }
    // rows
    const rowEls = {};
    for (const r of rows) {
      const re = h("div", { cls: "row" }, clip);
      const tint = h("div", { cls: "abs", style: { inset: "0", width: "100%", height: "100%" } }, re);
      const accent = h("div", { cls: "abs", style: { width: "3px", height: "100%", background: C.synth, borderRadius: "2px" } }, re);
      const line = h("div", { cls: "rowline" }, re);
      const cells = {};
      for (const c of cols) {
        const cell = h("div", { cls: "cell" + (c.kind === "num" || c.kind === "cat" || c.kind === "cand" || c.kind === "bool" ? " mono" : "") }, re);
        const v = r.vals[c.key];
        let inner;
        if (c.kind === "label" && v !== "…") {
          inner = h("span", { cls: "val1", text: String(v) }, cell);
          if (String(v) === "1") { inner.style.background = rgba(C.head, 0.16); inner.style.color = C.head; inner.style.border = `1px solid ${rgba(C.head, 0.45)}`; }
          else { inner.style.color = C.dim; inner.style.border = `1px solid ${C.line2}`; }
        } else {
          inner = h("span", { html: v == null ? "" : String(v) }, cell);
          if (c.kind === "cand") { inner.style.color = C.paper; inner.style.fontWeight = 500; }
          if (c.kind === "bool" && String(v) === "0") inner.style.color = C.dim;
          if (v === "…") inner.style.color = C.dim;
        }
        let lock = null;
        if (r.synth && c.lockable) {
          lock = icon(cell, "lock", 14, C.synth, 1.75);
          lock.style.marginLeft = "8px";
          lock.style.flex = "none";
        }
        let para = null;
        if (r.synth && c.kind === "text") {
          // the paraphrased text: framing words in violet
          inner.innerHTML = "";
          para = { orig: h("span", { text: r.orig, style: { position: "absolute", left: "14px", right: "14px", overflow: "hidden", textOverflow: "ellipsis" } }, cell) };
          const pw = h("span", { style: { position: "absolute", left: "14px", right: "14px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" } }, cell);
          para.wrap = pw;
          para.ws = [];
          r.para.forEach(([txt, isNew]) => {
            const s = h("span", { text: txt, style: { color: isNew ? C.synth : C.text2, fontWeight: isNew ? 600 : 400 } }, pw);
            if (isNew) Object.assign(s.style, { display: "inline-block", overflow: "hidden", verticalAlign: "top", whiteSpace: "pre" });
            para.ws.push({ s, isNew, full: 0 });
          });
        }
        cells[c.key] = { cell, inner, lock, para };
      }
      rowEls[r.id] = { re, tint, accent, line, cells, r };
    }
    return {
      root, frame, clip, rowEls, hcells, groups, colHl, rowH, headH, groupH,
      // st: {x,y,s,o, colW:{}, groupO, hdrAlt:0..1, rows:{id:{ins,o,fill,copy,para,tint}}, hl:{key:[color,k]},
      //      dimRepeat: 0..1, lock: 0..1}
      layout(st) {
        const colW = st.colW;
        const xs = {};
        let acc = 0;
        for (const c of cols) { xs[c.key] = acc; acc += colW[c.key]; }
        const W = acc;
        const gH = groupH * (st.groupO || 0);
        let y = gH + headH;
        const rowY = {};
        for (const r of rows) {
          const rs = st.rows[r.id] || { ins: 0 };
          rowY[r.id] = y;
          y += rowH * (rs.ins || 0);
        }
        return { xs, W, H: y, gH, rowY };
      },
      update(st) {
        const L = this.layout(st);
        const colW = st.colW;
        root.style.transformOrigin = "0 0";
        tf(root, { x: st.x, y: st.y, s: st.s || 1, o: st.o });
        place(frame, 0, 0, L.W, L.H);
        place(clip, 0, 0, L.W, L.H);
        // groups
        for (const k in groups) {
          const gg = groups[k];
          const keys = cols.filter((c) => c.group === k).map((c) => c.key);
          const gx = Math.min(...keys.map((kk) => L.xs[kk]));
          const gw = keys.reduce((s, kk) => s + colW[kk], 0);
          place(gg.gl, gx, 0, gw, Math.max(0, L.gH - 2));
          vis(gg.gl, (st.groupO || 0) * (st.groupK ? st.groupK[k] ?? 1 : 1));
          gg.bar.style.transformOrigin = "0 50%";
          const gk = st.groupBar ? st.groupBar[k] ?? 1 : 1;
          gg.bar.style.transform = `scaleX(${gk.toFixed(3)})`;
        }
        // header
        place(head, 0, L.gH, L.W, headH);
        place(headLine, 0, L.gH + headH - 1, L.W);
        for (const c of cols) {
          const hc = hcells[c.key];
          hc.cell.style.left = L.xs[c.key] + "px";
          hc.cell.style.width = colW[c.key] + "px";
          const ok = clamp((colW[c.key] - 40) / 40);
          if (hc.b) {
            const m = st.hdrAlt || 0; // 1 = show alt label
            hc.a.style.opacity = (ok * (1 - m)).toFixed(3);
            hc.b.style.opacity = (ok * m).toFixed(3);
          } else hc.a.style.opacity = ok.toFixed(3);
          const hk = st.hl && st.hl[c.key];
          hc.cell.style.color = hk && hk[1] > 0.02 ? hk[0] : C.dim;
          // column highlight
          const hlEl = colHl[c.key];
          if (hk && hk[1] > 0.001) {
            place(hlEl, L.xs[c.key], L.gH, colW[c.key], L.H - L.gH);
            hlEl.style.background = rgba(hk[0], 0.09 * hk[1]);
            hlEl.style.boxShadow = `inset 0 0 0 1px ${rgba(hk[0], 0.45 * hk[1])}`;
            vis(hlEl, 1);
          } else vis(hlEl, 0);
        }
        // rows
        for (const r of rows) {
          const R = rowEls[r.id];
          const rs = st.rows[r.id] || { ins: 0 };
          const ins = rs.ins || 0;
          place(R.re, 0, L.rowY[r.id], L.W, rowH * ins);
          vis(R.re, ins > 0.001 ? rs.o ?? 1 : 0);
          R.tint.style.background = r.synth ? rgba(C.synth, 0.07) : "transparent";
          vis(R.accent, r.synth ? 1 : 0);
          for (const c of cols) {
            const ce = R.cells[c.key];
            ce.cell.style.left = L.xs[c.key] + "px";
            ce.cell.style.width = colW[c.key] + "px";
            ce.cell.style.height = rowH + "px";
            const ok = clamp((colW[c.key] - 40) / 40);
            // reveal of a newly opened column's cell values
            const fillK = rs.fill && rs.fill[c.key] != null ? rs.fill[c.key] : 1;
            let o = ok * fillK;
            if (c.group === "dec" && r.repeat) o *= 1 - 0.45 * (st.dimRepeat || 0);
            if (r.synth && c.lockable) {
              const ck = rs.copy ?? 1;
              ce.inner.style.transform = `translateY(${(-rowH * 0.7 * (1 - ck)).toFixed(2)}px)`;
              o *= ck;
              if (ce.lock) ce.lock.style.opacity = ((st.lock || 0) * ck).toFixed(3);
            }
            if (ce.para) {
              const pk = rs.para ?? 1;
              ce.para.orig.style.opacity = (0).toFixed(3);
              const ck = rs.copy ?? 1;
              ce.para.ws.forEach((w) => {
                // the copied text arrives with the row; the new framing words then type in ahead of it
                if (w.isNew) {
                  if (!w.full) { w.s.style.maxWidth = "none"; w.full = w.s.scrollWidth; }
                  w.s.style.maxWidth = (w.full * E.out(pk)).toFixed(1) + "px";
                  w.s.style.opacity = Math.min(1, pk * 3).toFixed(3);
                } else w.s.style.opacity = ck.toFixed(3);
              });
            }
            ce.cell.style.opacity = o.toFixed(3);
          }
        }
        return L;
      },
    };
  }

  // ---------------------------------------------------------------- code editor with typing
  function CodeEditor(parent, x, y, w, lines, title) {
    const card = h("div", { cls: "abs card" }, parent);
    const H = 52 + lines.length * 30 + 22;
    place(card, x, y, w, H);
    const tab = h("div", {
      style: { position: "absolute", left: "18px", top: "14px", display: "flex", alignItems: "center", gap: "10px" },
    }, card);
    icon(tab, "code", 18, C.schema, 1.75);
    h("span", { cls: "mono", text: title, style: { fontSize: "15px", color: C.text2 } }, tab);
    h("div", { style: { position: "absolute", left: "0", right: "0", top: "46px", height: "1px", background: C.line } }, card);
    const CW = 17 * 0.6; // JetBrains Mono advance width at 17 px
    const ls = lines.map((ln, i) => {
      const row = h("div", { cls: "code", style: { position: "absolute", left: "18px", top: 58 + i * 30 + "px", height: "30px", overflow: "hidden" } }, card);
      h("span", { cls: "ln", text: String(i + 1) }, row);
      const body = h("span", { html: ln.html }, row);
      return { row, n: ln.text.length };
    });
    const caret = h("div", { style: { position: "absolute", width: "2px", height: "20px", background: C.schema, borderRadius: "1px" } }, card);
    const total = ls.reduce((s, l) => s + l.n, 0);
    return {
      card, H,
      update(o, k /* 0..1 typed */, dy = 0, caretOn = 1) {
        tf(card, { y: dy, o });
        let chars = Math.round(total * k);
        let caretSet = false;
        ls.forEach((l, i) => {
          const n = Math.max(0, Math.min(l.n, chars));
          chars -= l.n;
          l.row.style.width = 34 + n * CW + 2 + "px";
          if (!caretSet && (n < l.n || i === ls.length - 1)) {
            caret.style.left = 18 + 34 + n * CW + 1 + "px";
            caret.style.top = 58 + i * 30 + 5 + "px";
            caretSet = true;
          }
        });
        caret.style.opacity = (caretOn * (k > 0 ? 1 : 0)).toFixed(2);
      },
    };
  }

  g.TA = { C, rgba, ICON, icon, Mark, Wordmark, Background, Rail, PHASES, Narration, PhaseEyebrow, TextEyebrow, LLMBubble, TreeGlyph, Table, CodeEditor };
})(window);

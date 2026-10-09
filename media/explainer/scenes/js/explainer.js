// The TabAgent explainer (1920x1080, 73 s). Seven scenes on one timeline; script.json holds the narration.
(function (g) {
  "use strict";
  const { h, place, tf, vis, svgAttr, P, E, ENV, lerp, clamp, words, revealWords, pulse } = g.K;
  const { C, rgba, icon, Mark, Wordmark, Background, Rail, Narration, PhaseEyebrow, TextEyebrow, LLMBubble, TreeGlyph, Table, CodeEditor } = g.TA;

  const W = 1920, H = 1080;
  const TX = 120, TW = 590; // narration column
  const EY = 372, HY = 430; // eyebrow and headline tops
  const VX = 780; // visual area left edge (right edge 1800)

  // scene windows (seconds)
  const S = {
    hook: [0, 7.5], idea: [7.5, 15.3], schema: [15.3, 31.5], synth: [31.5, 40.0], head: [40.0, 52.0],
    payoff: [52.0, 63.0], cta: [63.0, 73.0],
  };

  function svgLayer(parent) {
    const s = h("svg", { cls: "abs", attrs: { width: W, height: H, viewBox: `0 0 ${W} ${H}` } }, parent);
    return s;
  }
  const path = (parent, d, stroke, sw = 1.5, extra = {}) =>
    h("path", { attrs: Object.assign({ d, fill: "none", stroke, "stroke-width": sw, "stroke-linecap": "round", "stroke-linejoin": "round" }, extra) }, parent);
  // draw a path progressively (k: 0..1); length is measured once
  function drawPath(p, k) {
    if (!p._len) { p._len = p.getTotalLength(); p.setAttribute("stroke-dasharray", p._len.toFixed(1)); }
    p.setAttribute("stroke-dashoffset", (p._len * (1 - k)).toFixed(1));
  }
  function pointAt(p, k) {
    if (!p._len) p._len = p.getTotalLength();
    return p.getPointAtLength(p._len * clamp(k));
  }
  function pill(parent, x, y, w, text, o = {}) {
    const el = h("div", { cls: "abs pill" }, parent);
    place(el, x, y, w);
    if (o.kind) h("span", { cls: "mono", text: o.kind, style: { fontSize: "13px", color: C.dim, letterSpacing: "0.04em", flex: "none" } }, el);
    else h("div", { cls: "dot", style: { background: o.dot || C.dim } }, el);
    const tx = h("span", { text, style: { overflow: "hidden", textOverflow: "ellipsis" } }, el);
    if (o.tag) h("span", { cls: "tag", text: o.tag }, el);
    el.tx = tx;
    return el;
  }
  function chip(parent, x, y, text, o = {}) {
    const el = h("div", { cls: "abs chip" }, parent);
    place(el, x, y);
    if (o.icon) icon(el, o.icon, 16, o.iconColor || C.dim, 1.75);
    el.tx = h("span", { text }, el);
    if (o.color) { el.style.color = o.color; el.style.borderColor = rgba(o.color, 0.45); el.style.background = rgba(o.color, 0.08); }
    return el;
  }
  function tagPill(parent, x, y, text, color, iconName) {
    const el = h("div", { cls: "abs tagpill" }, parent);
    place(el, x, y);
    el.style.color = color;
    el.style.border = `1px solid ${rgba(color, 0.45)}`;
    el.style.background = rgba(color, 0.08);
    if (iconName) icon(el, iconName, 18, color, 1.75);
    h("span", { text }, el);
    return el;
  }
  const centerX = (el, cx) => { el.style.left = cx - el.offsetWidth / 2 + "px"; };

  // ======================================================================================== build
  function build(stage, script, cfg) {
    stage.style.width = W + "px";
    stage.style.height = H + "px";
    const bg = Background(stage, W, H);
    const L = h("div", { cls: "abs", style: { width: W + "px", height: H + "px" } }, stage);
    const U = []; // per-frame updaters

    // ------------------------------------------------------------------ top bar
    const brand = h("div", { cls: "abs", style: { display: "flex", alignItems: "center", gap: "12px" } }, L);
    place(brand, TX, 50);
    Mark(brand, 36);
    const bt = h("div", { style: { fontWeight: 800, fontSize: "24px", letterSpacing: "-0.025em" } }, brand);
    h("span", { text: "Tab", style: { color: C.paper } }, bt);
    h("span", { text: "Agent", style: { background: `linear-gradient(90deg, ${C.schema}, ${C.synth} 52%, ${C.head})`, WebkitBackgroundClip: "text", color: "transparent" } }, bt);
    const rail = Rail(L, 1800, 49);
    U.push((t) => {
      tf(brand, { y: (1 - P(t, 0.1, 0.8)) * -10, o: P(t, 0.1, 0.8) * (1 - P(t, 62.7, 0.5, E.inOut)) });
      const ro = ENV(t, 14.5, 52.3, 0.7, 0.5);
      let act = 0;
      if (t >= 31.2) act = lerp(0, 1, P(t, 31.2, 0.6, E.inOut));
      if (t >= 39.7) act = lerp(1, 2, P(t, 39.7, 0.6, E.inOut));
      rail.update(t, ro, t < 15.0 ? lerp(-1, 0, P(t, 14.5, 0.5)) : act);
    });

    // ------------------------------------------------------------------ narration + eyebrows
    const colCues = script.cues.filter((c) => c.screen === "column");
    const narr = Narration(L, colCues, TX, HY, TW);
    U.push((t) => narr.update(t));
    const eyebrows = [
      [TextEyebrow(L, "The problem", TX, EY + 6), 0.25, 7.4],
      [TextEyebrow(L, "The idea", TX, EY + 6), 7.5, 15.2],
      [PhaseEyebrow(L, 0, TX, EY), 15.35, 31.4],
      [PhaseEyebrow(L, 1, TX, EY), 31.55, 39.9],
      [PhaseEyebrow(L, 2, TX, EY), 40.05, 51.9],
    ];
    U.push((t) => {
      for (const [el, a, b] of eyebrows) {
        const k = ENV(t, a, b, 0.55, 0.35);
        tf(el, { x: (1 - P(t, a, 0.55)) * -12, o: k });
      }
    });

    // ------------------------------------------------------------------ background glows
    U.push((t) => {
      const drift = (p) => Math.sin(t * 0.35 + p) * 30;
      bg.glow("llm", 1600 + drift(0), 470, 1000, 820, 0.85 * ENV(t, 3.9, 7.9, 1.2, 0.8) + 0.5 * ENV(t, 47.9, 52.0, 0.8, 0.6) + 0.55 * ENV(t, 52.2, 62.9, 1.0, 0.8));
      bg.glow("ok", 980 + drift(1), 470, 900, 700, 0.6 * ENV(t, 7.6, 13.4, 1.0, 0.8));
      bg.glow("schema", 1300 + drift(2), 520, 1300, 900, 0.85 * ENV(t, 14.8, 31.8, 1.2, 0.9));
      bg.glow("synth", 1280 + drift(3), 500, 1300, 900, 0.85 * ENV(t, 31.2, 40.4, 1.0, 0.9));
      bg.glow("head", 1340 + drift(4), 470, 1250, 900, 0.85 * ENV(t, 39.6, 52.4, 1.0, 0.9) + 0.5 * ENV(t, 52.2, 62.9, 1.0, 0.8));
      bg.glow("paper", 960, 470, 1500, 900, 0.7 * ENV(t, 63.0, 80, 1.4, 0.5));
      // the dot grid stays; it fades a touch under the end card
      vis(bg.dots, 1 - 0.4 * P(t, 63, 1.2));
    });

    buildHook(L, U);
    buildIdea(L, U);
    const card = buildCard(L);
    buildSchema(L, U, card);
    buildSynth(L, U, card);
    buildHead(L, U, card);
    buildPayoff(L, U, script);
    buildCTA(L, U, cfg, script);
    U.push((t) => card.frame(t));

    return { seek: (t) => { for (const u of U) u(t); } };
  }

  // ======================================================================================== 1 hook
  function buildHook(L, U) {
    const root = h("div", { cls: "layer" }, L);
    const svg = svgLayer(root);
    const label = TextEyebrow(root, "Agent run", VX + 10, 296);
    const PX = VX + 10, PW = 372, PY = 334, PG = 60;
    const steps = [
      ["User: return my roommate's order", C.dim],
      ["Read phone messages", C.dim],
      ["Extract item and instructions", C.dim],
    ].map(([t, d], i) => pill(root, PX, PY + i * PG, PW, t, { dot: d }));
    const dec = pill(root, PX, PY + 3 * PG, PW, "Shortlist Amazon APIs", { dot: C.paper, tag: "decision" });
    dec.style.fontWeight = 600;
    dec.style.color = C.paper;
    const qBadge = h("div", { cls: "abs badge", text: "?", style: { width: "28px", height: "28px", fontSize: "16px", background: C.paper } }, root);
    place(qBadge, PX + PW - 14, PY + 3 * PG - 12);

    const cands = ["show_orders", "initiate_return", "show_returns", "search_products", "add_to_cart", "cancel_order", "write_review", "…"];
    const CX0 = 1250, CY0 = 286, CG = 47;
    const chips = cands.map((c, i) => chip(root, CX0, CY0 + i * CG, c));
    const decY = PY + 3 * PG + 23;
    const fan = cands.map((c, i) => {
      const y = CY0 + i * CG + 18;
      return path(svg, `M${PX + PW + 4} ${decY} C ${PX + PW + 44} ${decY}, ${CX0 - 44} ${y}, ${CX0 - 6} ${y}`, "rgba(230,233,242,0.28)", 1.5);
    });
    const BX = 1648, BY = 452;
    const bubble = LLMBubble(root, BX, BY, 230);
    const conv = [];
    const tokens = [];
    const llmLabel = h("div", { cls: "abs", text: "frontier LLM", style: { fontWeight: 600, fontSize: "21px", color: C.llm } }, root);
    const track = h("div", { cls: "abs", style: { height: "4px", borderRadius: "2px", background: rgba(C.llm, 0.18) } }, root);
    const fill = h("div", { style: { height: "100%", width: "0%", borderRadius: "2px", background: C.llm } }, track);
    place(track, BX - 100, 616, 200);
    const tSlow = tagPill(root, 0, 662, "slow", C.llm, "clock");
    const tCost = tagPill(root, 0, 662, "costly", C.llm, "coin");
    // measure after fonts load
    let measured = false;
    U.push((t) => {
      if (!measured) {
        chips.forEach((c, i) => {
          const y = CY0 + i * CG + 18;
          const x0 = CX0 + c.offsetWidth + 6;
          const p = path(svg, `M${x0} ${y} C ${x0 + 60} ${y}, ${BX - 190} ${BY - 4}, ${BX - 118} ${BY - 4}`, rgba(C.llm, 0.4), 1.5);
          conv.push(p);
          for (let j = 0; j < 2; j++) tokens.push({ p, ph: (i * 0.37 + j * 0.5) % 1, el: h("circle", { attrs: { r: 3, fill: C.llm } }, svg) });
        });
        centerX(llmLabel, BX); llmLabel.style.top = "576px";
        const tw = tSlow.offsetWidth + 12 + tCost.offsetWidth;
        tSlow.style.left = BX - tw / 2 + "px";
        tCost.style.left = BX - tw / 2 + tSlow.offsetWidth + 12 + "px";
        measured = true;
      }
      const out = 1 - P(t, 6.95, 0.5, E.inOut);
      const on = t < 8.2;
      vis(root, on ? 1 : 0);
      if (!on) return;
      tf(label, { o: P(t, 0.45, 0.6) * out });
      steps.concat([dec]).forEach((s, i) => {
        const k = P(t, 0.55 + i * 0.12, 0.6);
        tf(s, { y: (1 - k) * 18 - 10 * P(t, 7.0, 0.5, E.inOut), o: k * (1 - P(t, 7.0, 0.5, E.inOut)) });
      });
      const dk = P(t, 1.25, 0.5);
      dec.style.borderColor = rgba(C.paper, 0.16 + 0.5 * dk);
      dec.style.boxShadow = `0 0 0 ${(4 * dk).toFixed(2)}px ${rgba(C.paper, 0.06 * dk)}`;
      tf(qBadge, { s: 0.6 + 0.4 * P(t, 1.3, 0.45, E.snap), o: P(t, 1.3, 0.3) * out });
      fan.forEach((p, i) => { drawPath(p, P(t, 1.55 + i * 0.04, 0.6)); p.setAttribute("opacity", out.toFixed(3)); });
      chips.forEach((c, i) => {
        const k = P(t, 1.7 + i * 0.06, 0.55);
        tf(c, { x: (1 - k) * -14, o: k * out });
      });
      // the LLM, deliberately slow
      const bk = P(t, 3.95, 0.9);
      bubble.update(t, bk * out, t > 4.6, 0.94 + 0.06 * bk);
      conv.forEach((p, i) => { drawPath(p, P(t, 4.15 + i * 0.07, 1.1, E.slow)); p.setAttribute("opacity", out.toFixed(3)); });
      tokens.forEach((tk) => {
        const k0 = P(t, 4.9, 0.6);
        const k = ((t - 4.9) / 2.6 + tk.ph) % 1;
        const pt = pointAt(tk.p, k);
        svgAttr(tk.el, { cx: pt.x.toFixed(1), cy: pt.y.toFixed(1), opacity: (k0 * out * Math.sin(Math.PI * k)).toFixed(3) });
      });
      tf(llmLabel, { o: P(t, 4.3, 0.6) * out });
      vis(track, P(t, 4.6, 0.5) * out);
      fill.style.width = (100 * 0.46 * P(t, 4.6, 2.9, E.lin)).toFixed(2) + "%";
      tf(tSlow, { y: (1 - P(t, 5.3, 0.6)) * 12, o: P(t, 5.3, 0.6) * out });
      tf(tCost, { y: (1 - P(t, 5.75, 0.6)) * 12, o: P(t, 5.75, 0.6) * out });
    });
  }

  // ======================================================================================== the card
  // One table instance carries the story: decision table (idea) -> feature card (TabSchema) -> paraphrased
  // rows (TabSynth) -> training rows (TabHead).
  const INTENT = "My roommate asked me to return something I had ordered for them and venmo them the money for it.";
  function buildCard(L) {
    const cols = [
      { key: "intent", label: "intent", alt: "context", kind: "text", group: "dec" },
      { key: "step_id", label: "step_id", kind: "num", group: "dec", lockable: true },
      { key: "app_name", label: "app_name", kind: "cat", group: "dec", lockable: true },
      { key: "candidate", label: "candidate", kind: "cand", group: "cand", lockable: true },
      { key: "mentioned_by_user", label: "mentioned_by_user", kind: "bool", group: "cand", lockable: true },
      { key: "args_in_context", label: "args_in_context", kind: "bool", group: "cand", lockable: true },
      { key: "used", label: "used", kind: "label", group: "label", lockable: true },
    ];
    const dec = { intent: INTENT, step_id: 24, app_name: "amazon" };
    const R = (id, cand, m, a, u, extra = {}) => Object.assign({ id, vals: Object.assign({}, dec, { candidate: cand, mentioned_by_user: m, args_in_context: a, used: u }) }, extra);
    const para = (prefix) => [[prefix, true], [" My roommate asked me to return something I had ordered for them and venmo them the money for it.", false]];
    const rows = [
      R("r0", "show_orders", 1, 1, 1),
      R("s0a", "show_orders", 1, 1, 1, { synth: true, para: para("Please handle the following."), orig: INTENT }),
      R("s0b", "show_orders", 1, 1, 1, { synth: true, para: para("Task:"), orig: INTENT }),
      R("r1", "initiate_return", 1, 1, 1, { repeat: true }),
      R("s1a", "initiate_return", 1, 1, 1, { synth: true, para: para("Please handle the following."), orig: INTENT }),
      R("r2", "show_returns", 1, 0, 0, { repeat: true }),
      R("r3", "search_products", 0, 0, 0, { repeat: true }),
      { id: "r4", vals: { intent: "…", step_id: "…", app_name: "…", candidate: "…", mentioned_by_user: "…", args_in_context: "…", used: "…" } },
    ];
    for (const r of rows) if (r.synth) r.vals.intent = INTENT;
    const tbl = Table(L, {
      cols, rows, rowH: 50, headH: 44, groupH: 38,
      groups: [
        { key: "dec", label: "per-decision features", color: C.schema },
        { key: "cand", label: "per-candidate features", color: C.schema },
        { key: "label", label: "label", color: C.head },
      ],
    });
    const realRows = ["r0", "r1", "r2", "r3", "r4"];
    // widths
    const Wnarrow = { intent: 250, step_id: 0, app_name: 0, candidate: 186, mentioned_by_user: 0, args_in_context: 0, used: 80 };
    const Wfull = { intent: 236, step_id: 88, app_name: 108, candidate: 180, mentioned_by_user: 172, args_in_context: 160, used: 76 };
    const Wsynth = { intent: 440, step_id: 96, app_name: 0, candidate: 202, mentioned_by_user: 190, args_in_context: 0, used: 78 };
    const mix = (A, B, k) => { const o = {}; for (const key in A) o[key] = lerp(A[key], B[key], k); return o; };

    // extra pieces drawn around the card
    const note = h("div", { cls: "abs small", style: { display: "flex", alignItems: "center", gap: "8px", fontSize: "16px" } }, L);
    icon(note, "table", 18, C.schema, 1.6);
    h("span", { text: "one decision, one row per candidate" }, note);
    const synthTag = h("div", { cls: "abs tagpill" }, L);
    synthTag.style.color = C.synth; synthTag.style.border = `1px solid ${rgba(C.synth, 0.45)}`; synthTag.style.background = rgba(C.synth, 0.08);
    icon(synthTag, "copy", 18, C.synth, 1.75);
    h("span", { text: "free text, paraphrased" }, synthTag);
    const lockTag = h("div", { cls: "abs tagpill" }, L);
    lockTag.style.color = C.text2; lockTag.style.border = `1px solid ${C.line2}`; lockTag.style.background = "rgba(255,255,255,0.03)";
    icon(lockTag, "lock", 18, C.synth, 1.75);
    h("span", { text: "copied verbatim" }, lockTag);
    // the training-set bar (TabSynth)
    const ts = h("div", { cls: "abs", style: { display: "flex", alignItems: "center", gap: "16px" } }, L);
    h("div", { cls: "eyebrow", text: "Training set", style: { fontSize: "15px" } }, ts);
    const tsBar = h("div", { style: { position: "relative", height: "14px", width: "500px" } }, ts);
    const tsReal = h("div", { style: { position: "absolute", left: "0", top: "0", height: "14px", borderRadius: "7px", background: "rgba(230,233,242,0.55)" } }, tsBar);
    const tsSyn = h("div", { style: { position: "absolute", top: "0", height: "14px", borderRadius: "7px", background: C.synth } }, tsBar);
    const tsLeg = h("div", { style: { display: "flex", gap: "18px", fontSize: "15px", color: C.dim, fontWeight: 500 } }, ts);
    const lg1 = h("span", { html: `<span style="display:inline-block;width:10px;height:10px;border-radius:3px;background:rgba(230,233,242,0.55);margin-right:7px"></span>real rows` }, tsLeg);
    const lg2 = h("span", { html: `<span style="display:inline-block;width:10px;height:10px;border-radius:3px;background:${C.synth};margin-right:7px"></span>paraphrased rows` }, tsLeg);

    const st = { rows: {} };
    return {
      tbl, realRows, Wnarrow, Wfull, Wsynth, mix, note, synthTag, lockTag, ts, tsReal, tsSyn,
      // computed per frame by the scenes, rendered last
      frame(t) {
        const s = this.state(t);
        if (!s) { vis(tbl.root, 0); vis(note, 0); vis(synthTag, 0); vis(lockTag, 0); vis(ts, 0); return; }
        const Lout = tbl.update(s);
        // annotations
        const sc = s.s || 1;
        place(note, s.x, s.y + Lout.H * sc + 20);
        vis(note, s.noteO || 0);
        place(synthTag, s.x + 6, s.y - 52);
        tf(synthTag, { y: (1 - (s.synthTagO || 0)) * 10, o: s.synthTagO || 0 });
        place(lockTag, s.x + Lout.xs.candidate + 4, s.y - 52);
        tf(lockTag, { y: (1 - (s.lockTagO || 0)) * 10, o: s.lockTagO || 0 });
        place(ts, s.x, s.y + Lout.H * sc + 34);
        vis(ts, s.tsO || 0);
        const r = 150, sy = (s.tsK || 0) * 300;
        tsReal.style.width = r + "px";
        tsSyn.style.left = r + 6 + "px";
        tsSyn.style.width = Math.max(0, sy) + "px";
        tsSyn.style.opacity = (s.tsK > 0.01 ? 1 : 0).toString();
      },
      state: () => null, // replaced below by a merged timeline
    };
  }

  // ======================================================================================== 2 idea
  function buildIdea(L, U) {
    const root = h("div", { cls: "layer" }, L);
    const svg = svgLayer(root);
    const lab = h("div", { cls: "abs", style: { display: "flex", alignItems: "center", gap: "10px" } }, root);
    place(lab, VX + 10, 296);
    const ck = h("div", { cls: "badge", style: { width: "22px", height: "22px", background: C.ok } }, lab);
    icon(ck, "check", 16, C.ink, 2.4);
    h("div", { cls: "eyebrow", text: "Successful runs" }, lab);
    const RX = VX + 10, RW = 64, RG = 18, RY = 340;
    const runs = [];
    for (let i = 0; i < 5; i++) {
      const x = RX + i * (RW + RG);
      const g0 = h("div", { cls: "abs" }, root);
      place(g0, x, RY, RW, 260);
      const badge = h("div", { cls: "badge", style: { position: "absolute", left: RW / 2 - 11 + "px", top: "0px", width: "22px", height: "22px", background: rgba(C.ok, 0.18), border: `1px solid ${rgba(C.ok, 0.6)}` } }, g0);
      icon(badge, "check", 14, C.ok, 2.2);
      const ps = [];
      for (let j = 0; j < 4; j++) {
        const p = h("div", { style: { position: "absolute", left: "0px", top: 36 + j * 34 + "px", width: RW + "px", height: "26px", borderRadius: "8px", background: C.ink2, border: `1px solid ${j === 3 ? rgba(C.paper, 0.6) : C.line2}` } }, g0);
        ps.push(p);
      }
      // the tools it chose (amber)
      const a1 = h("div", { style: { position: "absolute", left: "8px", top: 36 + 4 * 34 + "px", width: "30px", height: "7px", borderRadius: "4px", background: C.head } }, g0);
      const a2 = h("div", { style: { position: "absolute", left: "42px", top: 36 + 4 * 34 + "px", width: "14px", height: "7px", borderRadius: "4px", background: C.head } }, g0);
      runs.push({ g0, ps, a1, a2, cx: x + RW / 2, cy: RY + 36 + 3 * 34 + 13 });
    }
    // "the answers": the tools each successful run used at its decision
    const ans = h("div", { cls: "abs", style: { display: "flex", alignItems: "center", gap: "10px" } }, root);
    const ansLine = h("div", { style: { width: "2px", height: "22px", borderRadius: "1px", background: C.head } }, ans);
    h("span", { text: "the tools each run used", style: { fontWeight: 600, fontSize: "17px", color: C.head, whiteSpace: "nowrap" } }, ans);
    place(ans, RX + 4, RY + 36 + 4 * 34 + 18);
    const BLOCK = 5 * RW + 4 * RG;
    const SHIFT = 1290 - (RX + BLOCK / 2); // runs start centred in the visual area
    const ghosts = [0, 1, 2, 3, 4].map(() => h("div", { cls: "abs", style: { width: "44px", height: "10px", borderRadius: "5px", background: rgba(C.paper, 0.75), boxShadow: `0 0 18px ${rgba(C.paper, 0.35)}` } }, root));
    const tree = TreeGlyph(root, 1520, 312, 230);
    const tlab = h("div", { cls: "abs", text: "classifier", style: { fontWeight: 600, fontSize: "21px", color: C.head } }, root);
    const arrow = path(svg, "M1336 432 H 1492", rgba(C.paper, 0.5), 1.5);
    const arrowHead = path(svg, "M1484 425 L1493 432 L1484 439", rgba(C.paper, 0.5), 1.5);
    let measured = false;
    U.push((t) => {
      if (!measured) { centerX(tlab, 1520 + 115); tlab.style.top = "516px"; measured = true; }
      const on = t > 7.3 && t < 15.9;
      vis(root, on ? 1 : 0);
      if (!on) return;
      const runsOut = 1 - P(t, 12.9, 0.5, E.inOut);
      const dx = SHIFT * (1 - P(t, 10.9, 0.8, E.inOut));
      tf(lab, { x: dx, o: P(t, 7.5, 0.6) * runsOut });
      tf(ans, { x: dx, y: (1 - P(t, 9.3, 0.5)) * 8, o: P(t, 9.3, 0.5) * (1 - P(t, 11.2, 0.4)) });
      runs.forEach((r, i) => {
        const k = P(t, 7.5 + i * 0.08, 0.65);
        tf(r.g0, { x: dx, y: (1 - k) * 20, o: k * runsOut });
        const ak = P(t, 9.0 + i * 0.08, 0.4, E.snap);
        r.a1.style.transform = `scaleX(${ak.toFixed(3)})`; r.a1.style.transformOrigin = "0 50%";
        r.a2.style.transform = `scaleX(${ak.toFixed(3)})`; r.a2.style.transformOrigin = "0 50%";
        r.ps[3].style.boxShadow = `0 0 0 ${(3 * P(t, 9.6, 0.5)).toFixed(2)}px ${rgba(C.paper, 0.08)}`;
      });
      // rows fly from the runs' decisions into the table
      ghosts.forEach((gh, i) => {
        const a = 11.45 + i * 0.14;
        const k = P(t, a, 0.55, E.inOut);
        const r = runs[i];
        const tx = 1262 + 10, ty = 330 + 44 + i * 50 + 20;
        const x = lerp(r.cx + dx - 22, tx, k), y = lerp(r.cy - 5, ty, k) - Math.sin(Math.PI * k) * 40;
        tf(gh, { x, y, o: (k > 0 && k < 1 ? 1 : 0) * Math.sin(Math.PI * k) });
      });
      // the classifier
      const gk = P(t, 13.45, 0.9, E.lin);
      const out = 1 - P(t, 14.95, 0.5, E.inOut);
      tree.update(out * P(t, 13.4, 0.3), gk, [], 0);
      tf(tlab, { y: (1 - P(t, 13.9, 0.5)) * 10, o: P(t, 13.9, 0.5) * out });
      drawPath(arrow, P(t, 13.3, 0.5));
      arrow.setAttribute("opacity", out.toFixed(3));
      arrowHead.setAttribute("opacity", (P(t, 13.65, 0.25) * out).toFixed(3));
    });
  }

  // ======================================================================================== 3 TabSchema
  function buildSchema(L, U, card) {
    const root = h("div", { cls: "layer" }, L);
    const svg = svgLayer(root);
    // the four stages
    const NODES = [
      ["scan", "Analyze", "propose features"],
      ["code", "Write code", "extraction functions"],
      ["shield", "Validate", "sandbox and checks"],
      ["scale", "Judge", "3 judges + MetaJudge"],
    ];
    const NW = 242, NG = 17, NY = 192, NH = 78;
    const WY = 318; // workspace top
    const nodes = NODES.map(([ic, title, sub], i) => {
      const n = h("div", { cls: "abs card" }, root);
      place(n, VX + i * (NW + NG), NY, NW, NH);
      const sq = h("div", { style: { position: "absolute", left: "16px", top: "19px", width: "40px", height: "40px", borderRadius: "10px", display: "flex", alignItems: "center", justifyContent: "center", background: rgba(C.schema, 0.1) } }, n);
      const ico = icon(sq, ic, 22, C.schema, 1.6);
      h("div", { text: title, style: { position: "absolute", left: "70px", top: "16px", fontWeight: 600, fontSize: "19px", color: C.paper } }, n);
      h("div", { text: sub, style: { position: "absolute", left: "70px", top: "43px", fontWeight: 500, fontSize: "14.5px", color: C.dim, whiteSpace: "nowrap" } }, n);
      const done = h("div", { cls: "badge", style: { position: "absolute", right: "12px", top: "12px", width: "20px", height: "20px", background: C.schema } }, n);
      icon(done, "check", 14, C.ink, 2.4);
      return { n, sq, ico, done };
    });
    const conns = [0, 1, 2].map((i) => {
      const x = VX + (i + 1) * (NW + NG) - NG;
      return path(svg, `M${x + 3} ${NY + NH / 2} H ${x + NG - 3}`, rgba(C.schema, 0.6), 1.5);
    });
    // which stage is active (float), per time
    const activeAt = (t) => {
      if (t < 19.1) return 0;
      if (t < 21.1) return 1;
      if (t < 23.1) return 2;
      if (t < 26.9) return 3;
      return 4;
    };

    // ---- A: analyze
    const A = h("div", { cls: "layer" }, root);
    const tcard = h("div", { cls: "abs card" }, A);
    place(tcard, VX, WY, 430, 330);
    const thead = h("div", { style: { position: "absolute", left: "18px", top: "16px", display: "flex", alignItems: "center", gap: "10px" } }, tcard);
    const okb = h("div", { cls: "badge", style: { width: "20px", height: "20px", background: C.ok } }, thead);
    icon(okb, "check", 14, C.ink, 2.4);
    h("div", { cls: "eyebrow", text: "Successful run", style: { fontSize: "14px" } }, thead);
    const steps = [
      ["user", "My roommate asked me to return something…"],
      ["thought", "Extraction is complete: item and instructions…"],
      ["status", "Return request saved; next: Amazon"],
      ["step 24", "Shortlist Amazon APIs"],
    ];
    const SP = steps.map(([k, txt], i) => {
      const p = pill(tcard, 16, 56 + i * 64, 398, txt, { kind: k });
      p.style.height = "50px";
      p.style.position = "absolute";
      if (i === 3) { p.style.borderColor = rgba(C.paper, 0.55); p.tx.style.color = C.paper; p.tx.style.fontWeight = 600; }
      p.tx.style.fontSize = "16px";
      return p;
    });
    const scan = h("div", { cls: "abs", style: { width: "404px", height: "56px", borderRadius: "13px", border: `1.5px solid ${rgba(C.schema, 0.85)}`, background: rgba(C.schema, 0.07), boxShadow: `0 0 24px ${rgba(C.schema, 0.25)}` } }, A);
    const gDec = TextEyebrow(A, "Per decision", 1250, WY + 10, C.schema);
    const gCand = TextEyebrow(A, "Per candidate", 1250, WY + 186, C.schema);
    gDec.style.fontSize = gCand.style.fontSize = "15px";
    // proposed features: [name, group, source step, time]
    const FEATS = [
      ["intent", "dec", 0, 16.15], ["mentioned_by_user", "cand", 0, 16.35],
      ["raw_log_length", "dec", 1, 16.75], ["args_in_context", "cand", 1, 16.95],
      ["step_id", "dec", 3, 17.55], ["app_name", "dec", 3, 17.75],
    ];
    const slotsDec = [[1250, WY + 46], [1250 + 112, WY + 46], [1250, WY + 98], [1250 + 196, WY + 98]];
    const slotsCand = [[1250, WY + 222], [1250, WY + 274]];
    let di = 0, ci = 0;
    const fchips = FEATS.map(([name, grp, src, ta]) => {
      const c = chip(A, 0, 0, name, { color: C.schema });
      const slot = grp === "dec" ? slotsDec[di++] : slotsCand[ci++];
      return { c, name, slot, src, ta };
    });

    // ---- B: code + validate
    const B = h("div", { cls: "layer" }, root);
    const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
    const hl = (s) => esc(s)
      .replace(/("[^"]*")/g, "<span class='st'>$1</span>")
      .replace(/^(\s*)(def|return)\b/, "$1<span class='kw'>$2</span>")
      .replace(/(def<\/span> )(\w+)/, "$1<span class='fn'>$2</span>");
    const CODE = [
      "def extract_features_for_classification(trajectory_data):",
      "    steps = trajectory_data[\"steps\"]",
      "    return {\"step_id\": len(steps), \"app_name\": steps[-1][\"app\"]}",
      "",
      "def extract_candidate_features(trajectory_data, candidate):",
      "    request = trajectory_data[\"intent\"].lower()",
      "    words = candidate[\"description\"].lower().split()",
      "    return {\"mentioned_by_user\": any(w in request for w in words)}",
    ].map((s) => ({ text: s, html: hl(s) }));
    const editor = CodeEditor(B, VX, WY, 744, CODE, "extract.py");
    const vcard = h("div", { cls: "abs card" }, B);
    place(vcard, VX + 744 + 18, WY, 1800 - (VX + 744 + 18), editor.H);
    const vhead = h("div", { style: { position: "absolute", left: "18px", top: "14px", display: "flex", alignItems: "center", gap: "10px" } }, vcard);
    icon(vhead, "shield", 18, C.schema, 1.75);
    h("span", { text: "Sandbox run", style: { fontWeight: 600, fontSize: "16px", color: C.text2 } }, vhead);
    h("div", { style: { position: "absolute", left: "0", right: "0", top: "46px", height: "1px", background: C.line } }, vcard);
    const vtrack = h("div", { style: { position: "absolute", left: "18px", right: "18px", top: "68px", height: "5px", borderRadius: "3px", background: rgba(C.schema, 0.15) } }, vcard);
    const vfill = h("div", { style: { height: "100%", width: "0%", borderRadius: "3px", background: C.schema } }, vtrack);
    const CHECKS = ["runs in a sandbox", "reads only real fields", "no outcome fields"];
    const checks = CHECKS.map((txt, i) => {
      const row = h("div", { style: { position: "absolute", left: "18px", top: 98 + i * 48 + "px", display: "flex", alignItems: "center", gap: "12px" } }, vcard);
      const c = h("div", { cls: "badge", style: { width: "24px", height: "24px", border: `1.5px solid ${C.line2}`, background: "transparent" } }, row);
      const ic = icon(c, "check", 16, C.ink, 2.4);
      const lb = h("span", { text: txt, style: { fontWeight: 500, fontSize: "16px", color: C.text2 } }, row);
      return { row, c, ic, lb };
    });

    // ---- C: judges
    const Cc = h("div", { cls: "layer" }, root);
    const JX = [1118, 1288, 1458], MX = 1650;
    const JUDGES = ["Relevance", "Generality", "Impact & Uniqueness"];
    const jheads = JUDGES.map((name, j) => {
      const w = h("div", { cls: "abs", style: { display: "flex", flexDirection: "column", alignItems: "center", gap: "7px" } }, Cc);
      const av = h("div", { cls: "badge", style: { width: "38px", height: "38px", border: `1.5px solid ${rgba(C.schema, 0.7)}`, background: rgba(C.schema, 0.08), color: C.schema, fontSize: "16px" }, text: name[0] }, w);
      h("div", { text: name, style: { fontWeight: 500, fontSize: "14.5px", color: C.text2, whiteSpace: "nowrap" } }, w);
      return w;
    });
    const mhead = h("div", { cls: "abs", style: { display: "flex", flexDirection: "column", alignItems: "center", gap: "7px" } }, Cc);
    h("div", { cls: "badge", text: "M", style: { width: "38px", height: "38px", background: C.schema, fontSize: "16px" } }, mhead);
    h("div", { text: "MetaJudge", style: { fontWeight: 600, fontSize: "14.5px", color: C.paper } }, mhead);
    const JF = [
      ["intent", [5, 4, 4], 1], ["step_id", [4, 5, 4], 1], ["app_name", [4, 4, 5], 1],
      ["raw_log_length", [2, 1, 2], 0], ["mentioned_by_user", [5, 4, 5], 1], ["args_in_context", [4, 4, 4], 1],
    ];
    const JY0 = WY + 102, JYG = 44;
    const jrows = JF.map(([name, sc, acc], i) => {
      const c = chip(Cc, VX, JY0 + i * JYG, name);
      c.style.height = "34px";
      const dots = sc.map((s, j) => [0, 1, 2, 3, 4].map((d) => h("div", { cls: "abs", style: { width: "9px", height: "9px", borderRadius: "50%", background: C.line2 } }, Cc)));
      const verdict = h("div", { cls: "abs tagpill", text: acc ? "accept" : "reject", style: { height: "30px", padding: "0 12px", fontSize: "14.5px" } }, Cc);
      if (acc) { verdict.style.color = C.schema; verdict.style.background = rgba(C.schema, 0.12); verdict.style.border = `1px solid ${rgba(C.schema, 0.5)}`; }
      else { verdict.style.color = C.dim; verdict.style.border = `1px solid ${C.line2}`; }
      const strike = h("div", { cls: "abs", style: { height: "1.5px", background: C.dim } }, Cc);
      return { c, sc, acc, dots, verdict, strike, name };
    });

    let measured = false;
    U.push((t) => {
      const on = t > 14.9 && t < 32.2;
      vis(root, on ? 1 : 0);
      if (!on) return;
      if (!measured) {
        jheads.forEach((w, j) => { w.style.left = JX[j] - w.offsetWidth / 2 + 30 + "px"; w.style.top = WY + "px"; });
        mhead.style.left = MX - mhead.offsetWidth / 2 + 40 + "px"; mhead.style.top = WY + "px";
        jrows.forEach((r, i) => {
          r.dots.forEach((ds, j) => ds.forEach((d, k) => place(d, JX[j] + 30 - 30 + k * 13, JY0 + i * JYG + 12)));
          place(r.verdict, MX + 40 - r.verdict.offsetWidth / 2, JY0 + i * JYG + 2);
          place(r.strike, VX + 10, JY0 + i * JYG + 17, r.c.offsetWidth - 20);
        });
        fchips.forEach((f) => place(f.c, 0, 0));
        measured = true;
      }
      const outAll = 1 - P(t, 31.3, 0.55, E.inOut);
      // nodes
      const act = activeAt(t);
      nodes.forEach((nd, i) => {
        const k = P(t, 15.45 + i * 0.08, 0.6);
        const isA = act === i || (act === 1 && i === 1) ? 1 : 0;
        const a = isA ? P(t, i === 0 ? 15.6 : i === 1 ? 19.1 : i === 2 ? 21.1 : 23.1, 0.4) : 0;
        const done = act > i ? 1 : 0;
        const dimD = t > 26.9 ? 1 - 0.25 * P(t, 26.9, 0.5) : 1;
        tf(nd.n, { y: (1 - k) * 16, o: k * outAll * dimD });
        nd.n.style.borderColor = a ? rgba(C.schema, 0.25 + 0.5 * a) : done ? rgba(C.schema, 0.28) : "rgba(230,233,242,0.09)";
        nd.n.style.boxShadow = a ? `0 0 0 ${(4 * a).toFixed(1)}px ${rgba(C.schema, 0.08)}, 0 0 40px ${rgba(C.schema, 0.18 * a)}` : "";
        nd.sq.style.background = rgba(C.schema, 0.08 + 0.12 * Math.max(a, done * 0.5));
        tf(nd.done, { s: 0.5 + 0.5 * P(t, [19.1, 21.1, 23.1, 26.9][i], 0.35, E.snap), o: done ? P(t, [19.1, 21.1, 23.1, 26.9][i], 0.3) : 0 });
      });
      conns.forEach((p, i) => { drawPath(p, P(t, 15.7 + i * 0.1, 0.4)); p.setAttribute("opacity", ((act > i ? 1 : 0.35) * outAll).toFixed(2)); });

      // A
      const aO = ENV(t, 15.5, 19.4, 0.6, 0.4);
      vis(A, aO);
      if (aO > 0) {
        tf(tcard, { y: (1 - P(t, 15.5, 0.6)) * 18 });
        SP.forEach((p, i) => tf(p, { o: P(t, 15.6 + i * 0.07, 0.5) }));
        const sk = P(t, 15.95, 1.65, E.inOut);
        const sy = lerp(WY + 56 - 3, WY + 56 + 3 * 64 - 3, sk);
        tf(scan, { x: VX + 13, y: sy, o: P(t, 15.9, 0.3) * (1 - P(t, 17.9, 0.4)) });
        const gk = P(t, 16.0, 0.5);
        tf(gDec, { o: gk }); tf(gCand, { o: gk });
        fchips.forEach((f) => {
          const k = P(t, f.ta, 0.6, E.inOut);
          const sx = VX + 200, syy = WY + 56 + f.src * 64 + 8;
          const x = lerp(sx, f.slot[0], k), y = lerp(syy, f.slot[1], k) - Math.sin(Math.PI * k) * 26;
          tf(f.c, { x, y, s: 0.85 + 0.15 * k, o: P(t, f.ta, 0.25) });
        });
      }
      // B
      const bO = ENV(t, 19.15, 23.35, 0.55, 0.4);
      vis(B, bO);
      if (bO > 0) {
        editor.update(1, P(t, 19.35, 2.2, E.lin), (1 - P(t, 19.15, 0.55)) * 16, t < 23.0 ? 1 : 0);
        tf(vcard, { y: (1 - P(t, 19.3, 0.55)) * 16 });
        vfill.style.width = (100 * P(t, 21.15, 0.7, E.out)).toFixed(1) + "%";
        checks.forEach((c, i) => {
          const k = P(t, 21.55 + i * 0.4, 0.35, E.snap);
          c.c.style.background = rgba(C.ok, k);
          c.c.style.borderColor = k > 0.5 ? C.ok : C.line2;
          c.ic.style.opacity = k.toFixed(2);
          c.lb.style.color = k > 0.5 ? C.paper : C.text2;
        });
      }
      // C
      const cO = ENV(t, 23.15, 27.0, 0.55, 0.35);
      vis(Cc, cO);
      if (cO > 0) {
        const hk = P(t, 23.2, 0.5);
        jheads.forEach((w) => tf(w, { y: (1 - hk) * 10, o: hk }));
        tf(mhead, { y: (1 - hk) * 10, o: hk });
        jrows.forEach((r, i) => {
          const rk = P(t, 23.25 + i * 0.06, 0.5);
                    tf(r.c, { x: (1 - rk) * -12, o: rk * (r.acc ? 1 : 1 - 0.6 * P(t, 25.75, 0.4)) });
          r.dots.forEach((ds, j) => ds.forEach((d, k) => {
            const lit = k < r.sc[j] ? P(t, 23.55 + j * 0.5 + i * 0.05 + k * 0.03, 0.18, E.snap) : 0;
            d.style.background = lit > 0.5 ? C.schema : C.line2;
            d.style.transform = `scale(${(1 + 0.25 * Math.sin(Math.PI * lit)).toFixed(3)})`;
            d.style.opacity = rk.toFixed(3);
          }));
          const vk = P(t, 25.3 + i * 0.09, 0.4, E.snap);
          tf(r.verdict, { s: 0.85 + 0.15 * vk, o: vk });
          r.strike.style.transformOrigin = "0 50%";
          tf(r.strike, { sx: r.acc ? 0 : P(t, 25.75, 0.4), o: r.acc ? 0 : 1 });
        });
      }
    });
  }

  // ======================================================================================== card timeline
  // (shared by idea, TabSchema, TabSynth and TabHead)
  function cardTimeline(card) {
    const { Wnarrow, Wfull, Wsynth, mix } = card;
    card.state = (t) => {
      if (t < 11.2 || t > 41.6) return null;
      if (t > 15.9 && t < 26.85) return null;
      const st = { rows: {}, hl: {} };
      let colW, x, y, s = 1, o = 1;
      const rowsIn = (ids, a, gap) => ids.forEach((id, i) => { st.rows[id] = { ins: P(t, a + i * gap, 0.45) }; });
      if (t < 16) {
        // idea: the decision table assembles row by row, then slides left to meet the classifier
        colW = Wnarrow;
        const mk = P(t, 13.1, 0.85, E.inOut);
        x = lerp(1262, VX + 10, mk);
        y = 330;
        st.hdrAlt = 1;
        rowsIn(card.realRows, 11.7, 0.14);
        o = P(t, 11.3, 0.4) * (1 - P(t, 15.0, 0.55, E.inOut));
        y += P(t, 15.0, 0.55, E.inOut) * 24;
      } else if (t < 31.6) {
        // TabSchema: the table returns and grows its feature columns
        const ok = P(t, 26.95, 0.5);
        const ck = P(t, 27.35, 1.0, E.inOut);
        colW = mix(Wnarrow, Wfull, ck);
        x = VX;
        y = 336 + (1 - ok) * 20;
        o = ok;
        st.hdrAlt = 1 - P(t, 27.35, 0.5);
        card.realRows.forEach((id, i) => {
          st.rows[id] = { ins: 1, fill: { step_id: P(t, 27.95 + i * 0.07, 0.4), app_name: P(t, 28.0 + i * 0.07, 0.4), mentioned_by_user: P(t, 28.05 + i * 0.07, 0.4), args_in_context: P(t, 28.1 + i * 0.07, 0.4) } };
        });
        const hk = ENV(t, 27.4, 30.6, 0.4, 0.8);
        for (const k of ["step_id", "app_name", "mentioned_by_user", "args_in_context"]) st.hl[k] = [C.schema, hk];
        st.groupO = P(t, 28.35, 0.5);
        st.groupBar = { dec: P(t, 28.5, 0.6), cand: P(t, 28.6, 0.6), label: P(t, 28.7, 0.6) };
        st.dimRepeat = P(t, 29.2, 0.6);
        st.noteO = P(t, 29.5, 0.6) * (1 - P(t, 31.2, 0.4));
        // into TabSynth
        const sk = P(t, 31.45, 0.9, E.inOut);
        if (t > 31.45) colW = mix(Wfull, Wsynth, sk);
        y = lerp(y, 236, sk);
      } else if (t < 40.05) {
        const sk = P(t, 31.45, 0.9, E.inOut);
        colW = mix(Wfull, Wsynth, sk);
        x = VX;
        y = lerp(336, 236, sk);
        st.groupO = 1;
        st.dimRepeat = 1;
        card.realRows.forEach((id) => { st.rows[id] = { ins: 1 }; });
        st.hl.intent = [C.synth, ENV(t, 31.9, 39.9, 0.6, 0.6)];
        const ins = (id, a) => { st.rows[id] = { ins: P(t, a, 0.45, E.inOut), copy: P(t, a + 0.3, 0.5), para: P(t, a + 0.5, 0.9, E.lin) }; };
        ins("s0a", 32.7); ins("s0b", 33.4); ins("s1a", 34.1);
        st.synthTagO = ENV(t, 32.2, 39.6, 0.5, 0.4);
        st.lock = P(t, 35.65, 0.5);
        st.lockTagO = ENV(t, 35.6, 39.6, 0.5, 0.4);
        const vk = ENV(t, 35.65, 39.8, 0.5, 0.6);
        st.hl.candidate = [C.paper, 0.55 * vk];
        st.hl.used = [C.head, 0.8 * vk];
        st.hl.step_id = [C.paper, 0.35 * vk];
        st.hl.mentioned_by_user = [C.paper, 0.35 * vk];
        st.tsO = ENV(t, 37.4, 40.2, 0.5, 0.4);
        st.tsK = P(t, 37.7, 1.3, E.out);
      } else {
        // TabHead: the card shrinks into the training rows
        colW = Wsynth;
        const k = P(t, 40.05, 0.8, E.inOut);
        x = lerp(VX, VX, k);
        y = lerp(236, 334, k);
        s = lerp(1, 0.36, k);
        st.groupO = 1;
        st.dimRepeat = 1;
        for (const id of ["r0", "s0a", "s0b", "r1", "s1a", "r2", "r3", "r4"]) st.rows[id] = { ins: 1, copy: 1, para: 1 };
        st.lock = 1 - k;
        st.hl.intent = [C.synth, 1 - k];
        o = 1 - P(t, 41.1, 0.5, E.inOut);
      }
      Object.assign(st, { colW, x, y, s, o });
      return st;
    };
    card.cardState = (t) => {
      const s = card.state(t);
      if (!s) return null;
      s.L = card.tbl.layout(s);
      return s;
    };
  }

  // ======================================================================================== 4 TabSynth
  function buildSynth(L, U, card) {
    cardTimeline(card); // the TabSynth scene is the card itself (rows, highlights, tags)
  }

  // ======================================================================================== 5 TabHead
  function buildHead(L, U, card) {
    const root = h("div", { cls: "layer" }, L);
    const svg = svgLayer(root);
    const trLab = TextEyebrow(root, "Training rows", VX, 292);
    // a static thumbnail that takes over from the shrinking card
    const thumb = h("div", { cls: "abs card" }, root);
    place(thumb, VX, 334, 360, 176);
    const thumbRows = [];
    for (let i = 0; i < 8; i++) {
      const synth = [1, 2, 4].includes(i);
      const r = h("div", { style: { position: "absolute", left: "14px", top: 16 + i * 19 + "px", height: "9px", width: 300 - (i * 37) % 90 + "px", borderRadius: "5px", background: synth ? rgba(C.synth, 0.85) : "rgba(230,233,242,0.45)" } }, thumb);
      thumbRows.push(r);
    }
    const TX0 = 1228, TY0 = 312, TWD = 228;
    const tree = TreeGlyph(root, TX0, TY0, TWD);
    const tcx = TX0 + TWD / 2;
    const cbLab = h("div", { cls: "abs", text: "CatBoost", style: { fontWeight: 700, fontSize: "24px", color: C.head, letterSpacing: "-0.01em" } }, root);
    const feed = path(svg, `M${VX + 372} 422 H ${TX0 - 18}`, rgba(C.paper, 0.35), 1.5);
    const feedHead = path(svg, `M${TX0 - 26} 415 L${TX0 - 17} 422 L${TX0 - 26} 429`, rgba(C.paper, 0.35), 1.5);
    const bits = [0, 1, 2, 3].map(() => h("div", { cls: "abs", style: { width: "26px", height: "8px", borderRadius: "4px", background: C.synth } }, root));
    // badges: milliseconds, CPU
    const bMs = tagPill(root, 0, 0, "milliseconds", C.head, "stopwatch");
    const bCpu = tagPill(root, 0, 0, "on a CPU", C.head, "cpu");
    // ranked candidates
    const RLX = 1530, RLY = 312, RLG = 50, BARW = 250;
    const rlLab = TextEyebrow(root, "Ranked candidates", RLX, 274);
    const CANDS = [
      ["search_products", 0.17], ["show_returns", 0.36], ["initiate_return", 0.9], ["write_review", 0.06], ["show_orders", 0.95], ["cancel_order", 0.09],
    ];
    const order = CANDS.map((c, i) => i).sort((a, b) => CANDS[b][1] - CANDS[a][1]);
    const rankOf = {}; order.forEach((ci, r) => { rankOf[ci] = r; });
    const items = CANDS.map(([nm, sc], i) => {
      const w = h("div", { cls: "abs" }, root);
      place(w, RLX, RLY + i * RLG, 270, 44);
      const name = h("div", { cls: "mono", text: nm, style: { position: "absolute", left: "0px", top: "0px", fontSize: "16px", color: C.text2 } }, w);
      const tr = h("div", { style: { position: "absolute", left: "0px", top: "28px", width: BARW + "px", height: "6px", borderRadius: "3px", background: "rgba(230,233,242,0.08)" } }, w);
      const bar = h("div", { style: { height: "100%", width: "0px", borderRadius: "3px", background: "rgba(230,233,242,0.45)" } }, tr);
      return { w, name, bar, sc, i };
    });
    const bracket = path(svg, "", C.head, 2);
    const slLab = h("div", { cls: "abs", text: "shortlist", style: { fontWeight: 600, fontSize: "16px", color: C.head } }, root);
    // decisions
    const mkDec = (txt) => {
      const d = h("div", { cls: "abs pill", style: { height: "40px", fontSize: "16px", borderColor: rgba(C.paper, 0.55), color: C.paper, fontWeight: 600 } }, root);
      h("div", { cls: "dot", style: { background: C.paper } }, d);
      h("span", { text: txt }, d);
      return d;
    };
    const d1 = mkDec("Shortlist Amazon APIs");
    const dz = [mkDec("next decision"), mkDec("next decision")];
    const zOut = [0, 1].map(() => {
      const z = h("div", { cls: "abs badge", style: { width: "26px", height: "26px", background: C.head } }, root);
      icon(z, "check", 16, C.ink, 2.4);
      return z;
    });
    // the confidence gate
    const GX = 1238, GY = 640, GW = 210, GH = 74;
    const gate = h("div", { cls: "abs card" }, root);
    place(gate, GX, GY, GW, GH);
    h("div", { text: "confidence", style: { position: "absolute", left: "16px", top: "11px", fontWeight: 600, fontSize: "16px", color: C.text2 } }, gate);
    const mTrack = h("div", { style: { position: "absolute", left: "16px", top: "44px", width: GW - 32 + "px", height: "8px", borderRadius: "4px", background: "rgba(230,233,242,0.09)" } }, gate);
    const mFill = h("div", { style: { height: "100%", width: "0%", borderRadius: "4px", background: C.head } }, mTrack);
    const mTick = h("div", { style: { position: "absolute", left: (GW - 32) * 0.6 + "px", top: "-6px", width: "2px", height: "20px", borderRadius: "1px", background: C.paper, opacity: 0.6 } }, mTrack);
    const gIn = path(svg, `M${tcx} ${TY0 + 246} V ${GY - 4}`, rgba(C.paper, 0.35), 1.5);
    const gUp = path(svg, `M${GX + GW + 4} ${GY + GH / 2} H ${1530}`, rgba(C.head, 0.7), 1.75);
    const gDown = path(svg, `M${GX + GW / 2} ${GY + GH + 4} V ${852} H ${1500}`, rgba(C.llm, 0.6), 1.75);
    const useLab = h("div", { cls: "abs", style: { display: "flex", alignItems: "center", gap: "10px" } }, root);
    const useCk = h("div", { cls: "badge", style: { width: "26px", height: "26px", background: C.head } }, useLab);
    icon(useCk, "check", 16, C.ink, 2.4);
    h("span", { text: "use the shortlist", style: { fontWeight: 600, fontSize: "18px", color: C.head } }, useLab);
    place(useLab, 1544, GY + GH / 2 - 13);
    const llm = LLMBubble(root, 1560, 852, 112);
    const llmLab = h("div", { cls: "abs", text: "defer to the LLM", style: { fontWeight: 600, fontSize: "18px", color: C.llm } }, root);
    place(llmLab, 1630, 840);
    const tokA = h("div", { cls: "abs", style: { width: "14px", height: "14px", borderRadius: "50%", background: C.head, boxShadow: `0 0 14px ${rgba(C.head, 0.6)}` } }, root);
    const tokB = h("div", { cls: "abs", style: { width: "14px", height: "14px", borderRadius: "50%", background: C.paper, boxShadow: `0 0 14px ${rgba(C.paper, 0.4)}` } }, root);

    let measured = false;
    U.push((t) => {
      const on = t > 39.9 && t < 52.6;
      vis(root, on ? 1 : 0);
      if (!on) return;
      if (!measured) {
        centerX(cbLab, tcx); cbLab.style.top = TY0 + 204 + "px";
        const bw = bMs.offsetWidth + 10 + bCpu.offsetWidth;
        place(bMs, tcx - bw / 2, TY0 + 252); place(bCpu, tcx - bw / 2 + bMs.offsetWidth + 10, TY0 + 252);
        measured = true;
      }
      const out = 1 - P(t, 51.6, 0.6, E.inOut);
      const gateK = P(t, 47.6, 0.6, E.inOut); // the scene recomposes for the gate
      // training rows thumbnail and its feed into the tree
      const thK = P(t, 40.75, 0.45);
      tf(thumb, { o: thK * (1 - 0.55 * gateK) * out });
      tf(trLab, { o: P(t, 40.6, 0.5) * (1 - 0.55 * gateK) * out });
      thumbRows.forEach((r, i) => { r.style.transformOrigin = "0 50%"; r.style.transform = `scaleX(${P(t, 40.8 + i * 0.03, 0.4).toFixed(3)})`; });
      drawPath(feed, P(t, 40.9, 0.45));
      feed.setAttribute("opacity", ((1 - 0.6 * gateK) * out).toFixed(3));
      feedHead.setAttribute("opacity", (P(t, 41.25, 0.2) * (1 - 0.6 * gateK) * out).toFixed(3));
      bits.forEach((b, i) => {
        const k = P(t, 41.0 + i * 0.12, 0.4, E.inOut);
        tf(b, { x: lerp(VX + 330, TX0 - 30, k), y: 418, o: Math.sin(Math.PI * k) * out });
      });
      // the tree trains (grows) quickly
      const grow = P(t, 41.0, 0.9, E.lin);
      // decision path lighting (snappy)
      let pth = [], hl = 0;
      const light = (a, p) => { if (t >= a && t < a + 0.5) { pth = p; hl = Math.max(hl, P(t, a, 0.12, E.snap) * (1 - P(t, a + 0.3, 0.2))); } };
      light(42.18, [0, 2, 6 - 1]); light(44.95, [0, 1, 4]); light(45.75, [0, 2, 6]); light(48.25, [0, 2, 5]); light(49.85, [0, 1, 3]);
      tree.update(P(t, 40.95, 0.3) * out, grow, pth, hl, 0);
      tf(cbLab, { y: (1 - P(t, 41.4, 0.5)) * 10, o: P(t, 41.4, 0.5) * out });
      // decision 1
      const d1k = P(t, 41.75, 0.3);
      const d1m = P(t, 41.95, 0.24, E.snap);
      tf(d1, { x: lerp(VX, TX0 - 20, d1m), y: lerp(590, 410, d1m), s: lerp(1, 0.6, d1m), o: d1k * (1 - P(t, 42.15, 0.12)) * out });
      // bars snap, then the list sorts by score
      const sortK = P(t, 42.85, 0.35, E.snap);
      const listO = P(t, 41.6, 0.5) * out * (1 - 0.5 * gateK);
      tf(rlLab, { o: listO });
      items.forEach((it) => {
        const bk = P(t, 42.42 + it.i * 0.03, 0.22, E.snap);
        it.bar.style.width = (BARW * it.sc * bk).toFixed(1) + "px";
        const top = rankOf[it.i] < 2 && sortK > 0.5;
        it.bar.style.background = top ? C.head : "rgba(230,233,242,0.45)";
        it.name.style.color = top ? C.paper : C.text2;
        const y = lerp(it.i * RLG, rankOf[it.i] * RLG, sortK);
        tf(it.w, { x: RLX, y: RLY + y, o: listO });
        it.w.style.left = "0px"; it.w.style.top = "0px";
      });
      const bkK = P(t, 43.3, 0.35);
      bracket.setAttribute("d", `M${RLX - 14} ${RLY + 2} H ${RLX - 20} V ${RLY + 2 * RLG - 8} H ${RLX - 14}`);
      drawPath(bracket, bkK);
      bracket.setAttribute("opacity", (out * (1 - 0.5 * gateK)).toFixed(3));
      slLab.style.left = RLX + BARW - slLab.offsetWidth + "px"; slLab.style.top = RLY + "px";
      tf(slLab, { o: bkK * out * (1 - 0.5 * gateK) });
      // milliseconds, on a CPU
      tf(bMs, { y: (1 - P(t, 44.1, 0.4, E.snap)) * 10, o: P(t, 44.1, 0.35) * out * (1 - gateK) });
      tf(bCpu, { y: (1 - P(t, 44.3, 0.4, E.snap)) * 10, o: P(t, 44.3, 0.35) * out * (1 - gateK) });
      // two more decisions zip through
      [[44.7, 0], [45.5, 1]].forEach(([a, i]) => {
        const m = P(t, a, 0.22, E.snap);
        tf(dz[i], { x: lerp(VX, TX0 - 20, m), y: lerp(590, 410, m), s: lerp(1, 0.6, m), o: P(t, a - 0.12, 0.12) * (1 - P(t, a + 0.2, 0.1)) * out });
        const zm = P(t, a + 0.28, 0.3, E.snap);
        tf(zOut[i], { x: lerp(TX0 + TWD + 6, TX0 + TWD + 90, zm), y: 400, s: 0.7 + 0.3 * zm, o: (t > a + 0.28 ? 1 : 0) * (1 - P(t, a + 0.55, 0.25)) * out });
      });
      // the gate
      tf(gate, { y: (1 - gateK) * 16, o: gateK * out });
      drawPath(gIn, P(t, 47.8, 0.4));
      gIn.setAttribute("opacity", out.toFixed(3));
      drawPath(gUp, P(t, 47.95, 0.4));
      gUp.setAttribute("opacity", out.toFixed(3));
      drawPath(gDown, P(t, 48.05, 0.5));
      gDown.setAttribute("opacity", out.toFixed(3));
      tf(useLab, { o: P(t, 48.1, 0.4) * out });
      llm.update(t, P(t, 48.2, 0.5) * out, t > 50.2, 1);
      tf(llmLab, { o: P(t, 48.25, 0.4) * out });
      // decision A: confident -> the shortlist
      const tokPath = (t0) => {
        // into the tree, down to the gate, then (after the meter) out of it
        const kIn = P(t, t0, 0.22, E.snap), kDown = P(t, t0 + 0.4, 0.2, E.snap);
        if (t < t0 + 0.4) return [lerp(VX + 20, tcx - 7, kIn), lerp(600, TY0 + 92, kIn), 1 - P(t, t0 + 0.24, 0.1)];
        return [tcx - 7, lerp(TY0 + 246, GY - 18, kDown), P(t, t0 + 0.4, 0.06) * (1 - P(t, t0 + 0.6, 0.12))];
      };
      let [ax, ay, ao] = tokPath(48.0);
      if (t > 48.95) { const k = P(t, 48.95, 0.28, E.snap); ax = lerp(GX + GW, 1526, k); ay = GY + GH / 2 - 7; ao = P(t, 48.95, 0.05); }
      tf(tokA, { x: ax, y: ay, o: ao * P(t, 47.95, 0.1) * (1 - P(t, 49.3, 0.2)) * out });
      // decision B: unsure -> back to the LLM (slow)
      let [bx, by, bo] = tokPath(49.6);
      if (t > 50.55) { const pt = pointAt(gDown, P(t, 50.55, 1.0, E.slow)); bx = pt.x - 7; by = pt.y - 7; bo = P(t, 50.55, 0.05); }
      tf(tokB, { x: bx, y: by, o: bo * P(t, 49.55, 0.1) * (1 - P(t, 51.5, 0.2)) * out });
      // the meter: A fills high, B fills low
      let mv = 0, mc = C.head;
      if (t > 48.5 && t < 49.5) mv = 0.86 * P(t, 48.55, 0.3, E.snap);
      if (t >= 49.5 && t < 50.1) mv = 0.86 * (1 - P(t, 49.5, 0.3));
      if (t >= 50.1) { mv = 0.3 * P(t, 50.15, 0.3, E.snap); mc = C.llm; }
      mFill.style.width = (100 * mv).toFixed(1) + "%";
      mFill.style.background = mc;
      useCk.style.transform = `scale(${(1 + 0.18 * Math.sin(Math.PI * P(t, 49.2, 0.35))).toFixed(3)})`;
    });
  }

  // ======================================================================================== 6 payoff
  function buildPayoff(L, U, script) {
    const root = h("div", { cls: "layer" }, L);
    const eb = TextEyebrow(root, "AppWorld tool shortlisting · per decision", TX, 150);
    const head = h("div", { cls: "abs", style: { fontWeight: 800, fontSize: "76px", lineHeight: 1.06, letterSpacing: "-0.025em", color: C.paper, whiteSpace: "nowrap" } }, root);
    place(head, TX - 4, 190);
    const l1 = h("div", {}, head);
    const l2 = h("div", {}, head);
    const mult = (parent, m, rest) => {
      const ws = [h("span", { cls: "w", html: m.replace("~", "<span style='position:relative;top:-0.12em'>~</span>"), style: { color: C.head } }, parent)];
      parent.appendChild(document.createTextNode(" "));
      return ws.concat(words(parent, rest));
    };
    const w1 = mult(l1, "~10³×", "lower latency.");
    const w2 = mult(l2, "~10⁵×", "lower cost.");
    // the race: latency and cost, the LLM vs the head
    const NX = 380, BX = 640, BW = 820;
    const rows = [];
    const mkRow = (label, y, llmVal, headVal, frac) => {
      const lab = h("div", { cls: "abs", text: label, style: { fontWeight: 600, fontSize: "22px", color: C.dim } }, root);
      place(lab, TX, y + 6);
      const nA = h("div", { cls: "abs", text: "GPT-5.4 shortlister", style: { fontWeight: 500, fontSize: "20px", color: C.llm } }, root);
      place(nA, NX, y);
      const nB = h("div", { cls: "abs", text: "TabAgent head", style: { fontWeight: 600, fontSize: "20px", color: C.head } }, root);
      place(nB, NX, y + 58);
      const tA = h("div", { cls: "abs", style: { height: "14px", borderRadius: "7px", background: rgba(C.llm, 0.12) } }, root);
      place(tA, BX, y + 6, BW);
      const fA = h("div", { style: { height: "100%", width: "0px", borderRadius: "7px", background: C.llm } }, tA);
      const tB = h("div", { cls: "abs", style: { height: "14px", borderRadius: "7px", background: rgba(C.head, 0.1) } }, root);
      place(tB, BX, y + 64, BW);
      const fB = h("div", { style: { height: "100%", width: "0px", borderRadius: "7px", background: C.head, boxShadow: `0 0 16px ${rgba(C.head, 0.7)}` } }, tB);
      const vA = h("div", { cls: "abs mono", html: llmVal, style: { fontWeight: 600, fontSize: "34px", color: C.llm } }, root);
      place(vA, BX + BW + 28, y - 8);
      const vB = h("div", { cls: "abs mono", html: headVal, style: { fontWeight: 700, fontSize: "34px", color: C.head } }, root);
      place(vB, BX + BW + 28, y + 50);
      rows.push({ lab, nA, nB, tA, fA, tB, fB, vA, vB, frac });
    };
    const sup = (s) => `<sup style="font-size:0.58em;vertical-align:0;position:relative;top:-0.62em;margin-left:1px">${s}</sup>`;
    mkRow("Latency", 410, "7.50 s", "2.682 ms", 2.682 / 7500);
    mkRow("Cost", 590, "$0.052", "$2.0×10" + sup("−7"), 2.0e-7 / 0.052);
    const acc = h("div", { cls: "abs card", style: { display: "flex", alignItems: "center", gap: "22px", padding: "0 30px" } }, root);
    place(acc, TX, 780, 1680, 112);
    const okc = h("div", { cls: "badge", style: { width: "44px", height: "44px", background: C.ok, flex: "none" } }, acc);
    icon(okc, "check", 28, C.ink, 2.6);
    h("div", { html: "Recovers essentially all of the tools the GPT-5.4 shortlister used on its successful runs <span style=\"color:#94A3B8;font-weight:500\">(Recall@9)</span>.", style: { fontWeight: 600, fontSize: "29px", letterSpacing: "-0.01em", color: C.paper, lineHeight: 1.25 } }, acc);
    const src = h("div", { cls: "abs small", text: "Source: the TabAgent paper (AppWorld tool shortlisting; latency and cost per decision).", style: { fontSize: "16px" } }, root);
    place(src, TX, 990);
    U.push((t) => {
      const on = t > 51.7 && t < 63.6;
      vis(root, on ? 1 : 0);
      if (!on) return;
      const out = 1 - P(t, 62.75, 0.5, E.inOut);
      tf(eb, { x: (1 - P(t, 52.1, 0.6)) * -12, o: P(t, 52.1, 0.6) * out });
      revealWords(w1, t, 52.3, { stagger: 0.07, dur: 0.6, dy: 26 });
      revealWords(w2, t, 52.6, { stagger: 0.07, dur: 0.6, dy: 26 });
      vis(head, out);
      rows.forEach((r, i) => {
        const a = 53.0 + i * 1.1;
        const ok = P(t, a - 0.25, 0.5);
        [r.lab, r.nA, r.nB].forEach((e) => tf(e, { x: (1 - ok) * -10, o: ok * out }));
        vis(r.tA, ok * out); vis(r.tB, ok * out);
        // the LLM crawls; the head snaps (its bar is a sliver at this scale)
        const kA = P(t, a, 2.3, E.slow);
        const kB = P(t, a, 0.16, E.snap);
        r.fA.style.width = (BW * kA).toFixed(1) + "px";
        r.fB.style.width = Math.max(6, BW * r.frac) * kB + "px";
        tf(r.vA, { x: (1 - P(t, a + 2.2, 0.4)) * -10, o: P(t, a + 2.2, 0.4) * out });
        tf(r.vB, { x: (1 - P(t, a + 0.12, 0.25, E.snap)) * -10, o: P(t, a + 0.12, 0.2) * out });
      });
      const ak = P(t, 57.0, 0.7);
      tf(acc, { y: (1 - ak) * 20, o: ak * out });
      tf(src, { o: P(t, 53.5, 0.8) * out });
    });
  }

  // ======================================================================================== 7 call to action
  function buildCTA(L, U, cfg, script) {
    const root = h("div", { cls: "layer" }, L);
    const wm = Wordmark(root, 108);
    const sub = h("div", { cls: "abs", text: "Distilling LLM Agent Components into Cheap Classifiers", style: { fontWeight: 500, fontSize: "32px", color: C.text2, letterSpacing: "-0.01em", whiteSpace: "nowrap" } }, root);
    const lead = h("div", { cls: "abs", text: "Onboard your own agent from its logs", style: { fontWeight: 600, fontSize: "26px", color: C.paper, whiteSpace: "nowrap" } }, root);
    const term = h("div", { cls: "abs card" }, root);
    const TW2 = 1000;
    place(term, (W - TW2) / 2, 590, TW2, 150);
    const tl = [cfg.install_cmd, cfg.onboard_cmd].map((cmd, i) => {
      const row = h("div", { cls: "mono", style: { position: "absolute", left: "34px", top: 30 + i * 50 + "px", height: "40px", fontSize: "27px", lineHeight: "40px", whiteSpace: "pre", overflow: "hidden" } }, term);
      h("span", { text: "$ ", style: { color: C.slate } }, row);
      const body = h("span", { html: colorCmd(cmd), style: { color: C.paper } }, row);
      return { row, n: cmd.length };
    });
    const caret = h("div", { style: { position: "absolute", width: "3px", height: "30px", background: C.head, borderRadius: "1px" } }, term);
    const url = h("div", { cls: "abs", style: { display: "flex", alignItems: "center", gap: "16px", whiteSpace: "nowrap" } }, root);
    icon(url, "arrowUR", 30, C.paper, 2);
    h("span", { text: cfg.repo_url, style: { fontWeight: 600, fontSize: "36px", color: C.paper, letterSpacing: "-0.015em" } }, url);
    h("span", { text: "·", style: { fontWeight: 500, fontSize: "30px", color: C.slate, margin: "0 4px" } }, url);
    h("span", { text: "arXiv " + cfg.arxiv_id, style: { fontWeight: 500, fontSize: "28px", color: C.dim } }, url);
    let measured = false;
    const CWm = 27 * 0.6;
    U.push((t) => {
      const on = t > 62.8;
      vis(root, on ? 1 : 0);
      if (!on) return;
      if (!measured) {
        wm.el.style.position = "absolute";
        place(wm.el, (W - wm.el.offsetWidth) / 2, 236);
        place(sub, (W - sub.offsetWidth) / 2, 404);
        place(lead, (W - lead.offsetWidth) / 2, 530);
        place(url, (W - url.offsetWidth) / 2, 800);
        measured = true;
      }
      wm.mark.assemble(P(t, 63.15, 1.1, E.lin));
      const tk = P(t, 63.45, 0.8);
      tf(wm.txt, { x: (1 - tk) * 18, o: tk });
      tf(sub, { y: (1 - P(t, 63.95, 0.7)) * 14, o: P(t, 63.95, 0.7) });
      tf(lead, { y: (1 - P(t, 67.4, 0.6)) * 14, o: P(t, 67.4, 0.6) });
      tf(term, { y: (1 - P(t, 67.55, 0.6)) * 16, o: P(t, 67.55, 0.6) });
      const ks = [P(t, 67.95, 0.85, E.lin), P(t, 68.95, 1.05, E.lin)];
      let caretSet = false;
      tl.forEach((l, i) => {
        const n = Math.round(l.n * ks[i]);
        l.row.style.width = (2 + n) * CWm + 2 + "px";
        l.row.style.opacity = ks[i] > 0 ? "1" : "0";
        if (!caretSet && (ks[i] < 1 || i === tl.length - 1)) {
          caret.style.left = 34 + (2 + n) * CWm + 2 + "px";
          caret.style.top = 30 + i * 50 + 5 + "px";
          caretSet = true;
        }
      });
      caret.style.opacity = (P(t, 67.9, 0.2)).toFixed(2);
      tf(url, { y: (1 - P(t, 70.1, 0.7)) * 14, o: P(t, 70.1, 0.7) });
    });
  }
  function colorCmd(cmd) {
    const esc = cmd.replace(/&/g, "&amp;").replace(/</g, "&lt;");
    return esc
      .replace(/"([^"]*)"/g, `<span style='color:#93A4BF'>"$1"</span>`)
      .replace(/(--[\w-]+)/g, `<span style='color:${C.schema}'>$1</span>`);
  }

  g.Explainer = { build, W, H, S };
})(window);

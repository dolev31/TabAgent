// The README hero loop (1280x720, 8 s, seamless): successful runs -> TabSchema -> TabSynth -> TabHead -> a
// ranked shortlist. Every moving part is periodic in t with a period that divides 8 s, so the frame at
// t = 8 is the frame at t = 0. The background never moves (small animated WebP frames).
(function (g) {
  "use strict";
  const { h, place, tf, vis, P, E, lerp, clamp, pulse } = g.K;
  const { C, rgba, icon, Mark, Background, TreeGlyph } = g.TA;

  const W = 1280, H = 640, PERIOD = 8, BEAT = 2; // a new item enters every BEAT seconds
  const mod = (x, m) => ((x % m) + m) % m;

  function build(stage, script, cfg) {
    stage.style.width = W + "px";
    stage.style.height = H + "px";
    const bg = Background(stage, W, H);
    bg.glow("schema", 350, 400, 620, 520, 0.55);
    bg.glow("synth", 630, 400, 620, 520, 0.5);
    bg.glow("head", 910, 400, 620, 520, 0.55);
    const L = h("div", { cls: "layer" }, stage);
    const U = [];

    // brand
    const brand = h("div", { cls: "abs", style: { display: "flex", alignItems: "center", gap: "10px" } }, L);
    place(brand, 48, 40);
    Mark(brand, 30);
    const bt = h("div", { style: { fontWeight: 800, fontSize: "21px", letterSpacing: "-0.025em" } }, brand);
    h("span", { text: "Tab", style: { color: C.paper } }, bt);
    h("span", { text: "Agent", style: { background: `linear-gradient(90deg, ${C.schema}, ${C.synth} 52%, ${C.head})`, WebkitBackgroundClip: "text", color: "transparent" } }, bt);
    const headline = h("div", { cls: "abs", text: "Replace an LLM agent's closed-set decisions with a cheap classifier.", style: { fontWeight: 700, fontSize: "33px", letterSpacing: "-0.022em", color: C.paper, whiteSpace: "nowrap" } }, L);
    place(headline, 48, 100);
    const sub = h("div", { cls: "abs", text: "Trained on the agent's own successful runs.", style: { fontWeight: 500, fontSize: "19px", color: C.dim, whiteSpace: "nowrap" } }, L);
    place(sub, 48, 148);

    const CY = 392; // pipeline centre line
    // ------------------------------------------------------------ the agent's logs
    const logs = h("div", { cls: "layer" }, L);
    for (let i = 2; i >= 0; i--) {
      const c = h("div", { cls: "abs card" }, logs);
      place(c, 48 + i * 10, CY - 92 + i * 10, 112, 150);
      for (let j = 0; j < 4; j++) h("div", { style: { position: "absolute", left: "14px", top: 40 + j * 24 + "px", height: "8px", width: [74, 60, 80, 52][j] + "px", borderRadius: "4px", background: j === 3 ? rgba(C.paper, 0.75) : "rgba(230,233,242,0.28)" } }, c);
      const ck = h("div", { cls: "badge", style: { position: "absolute", left: "14px", top: "12px", width: "18px", height: "18px", background: C.ok } }, c);
      icon(ck, "check", 12, C.ink, 2.6);
    }
    const logLab = h("div", { cls: "abs", html: `<div style="font-weight:600;font-size:17px;color:${C.paper}">Agent logs</div><div style="font-weight:500;font-size:14px;color:${C.dim};margin-top:3px">successful runs</div>` }, L);
    place(logLab, 48, CY + 96);

    // ------------------------------------------------------------ the three phases
    const NODES = [
      { x: 232, color: C.schema, n: 1, title: "TabSchema", sub: "discovers features" },
      { x: 512, color: C.synth, n: 2, title: "TabSynth", sub: "augments the rows" },
      { x: 792, color: C.head, n: 3, title: "TabHead", sub: "decides in milliseconds" },
    ];
    const NW = 236, NH = 220, NY = CY - 118;
    const nodes = NODES.map((nd) => {
      const c = h("div", { cls: "abs card" }, L);
      place(c, nd.x, NY, NW, NH);
      c.style.borderColor = rgba(nd.color, 0.45);
      h("div", { style: { position: "absolute", left: "18px", right: "18px", top: "0", height: "2px", borderRadius: "0 0 2px 2px", background: nd.color } }, c);
      h("div", { cls: "badge", text: String(nd.n), style: { position: "absolute", left: "18px", top: "20px", width: "24px", height: "24px", fontSize: "13px", background: nd.color } }, c);
      h("div", { text: nd.title, style: { position: "absolute", left: "18px", top: "54px", fontWeight: 700, fontSize: "25px", letterSpacing: "-0.015em", color: C.paper } }, c);
      h("div", { text: nd.sub, style: { position: "absolute", left: "18px", top: "88px", fontWeight: 600, fontSize: "15px", color: nd.color, whiteSpace: "nowrap" } }, c);
      const viz = h("div", { cls: "abs", style: { left: "18px", top: "124px", width: NW - 36 + "px", height: "78px" } }, c);
      return { c, viz, nd };
    });
    // TabSchema: a feature grid whose columns light one after another
    const grid = [];
    for (let r = 0; r < 3; r++) for (let k = 0; k < 5; k++) {
      const cell = h("div", { style: { position: "absolute", left: k * 40 + "px", top: 6 + r * 24 + "px", width: "34px", height: "14px", borderRadius: "4px", background: rgba(C.schema, 0.15) } }, nodes[0].viz);
      grid.push({ cell, r, k });
    }
    // TabSynth: a real row and its paraphrased copies
    const sy = [];
    for (let r = 0; r < 3; r++) {
      const row = h("div", { style: { position: "absolute", left: "0px", top: 6 + r * 24 + "px", width: "200px", height: "14px", borderRadius: "4px", background: r === 0 ? "rgba(230,233,242,0.55)" : C.synth } }, nodes[1].viz);
      sy.push(row);
    }
    // TabHead: the tree, lighting a path per decision
    const tree = TreeGlyph(nodes[2].viz, 60, -4, 96);

    // ------------------------------------------------------------ the decision
    const out = h("div", { cls: "abs card" }, L);
    place(out, 1072, CY - 92, 160, 150);
    const bars = [0, 1, 2, 3].map((i) => {
      const tr = h("div", { style: { position: "absolute", left: "14px", top: 40 + i * 24 + "px", width: "132px", height: "8px", borderRadius: "4px", background: "rgba(230,233,242,0.08)" } }, out);
      return h("div", { style: { height: "100%", width: "0px", borderRadius: "4px" } }, tr);
    });
    h("div", { cls: "eyebrow", text: "Decision", style: { position: "absolute", left: "14px", top: "12px", fontSize: "12px" } }, out);
    const outLab = h("div", { cls: "abs", html: `<div style="font-weight:600;font-size:17px;color:${C.paper}">Ranked shortlist</div><div style="font-weight:500;font-size:14px;color:${C.dim};margin-top:3px">in one classifier pass</div>` }, L);
    place(outLab, 1072, CY + 96);

    // ------------------------------------------------------------ connectors and moving items
    const svg = h("svg", { cls: "abs", attrs: { width: W, height: H, viewBox: `0 0 ${W} ${H}` } }, L);
    const SEG = [
      [180, 232, "rgba(230,233,242,0.5)"],
      [468, 512, C.schema],
      [748, 792, C.synth],
      [1028, 1072, C.head],
    ];
    SEG.forEach(([a, b, col]) => {
      h("path", { attrs: { d: `M${a + 4} ${CY} H ${b - 6}`, stroke: col, "stroke-opacity": 0.55, "stroke-width": 1.5, fill: "none", "stroke-linecap": "round" } }, svg);
      h("path", { attrs: { d: `M${b - 12} ${CY - 5} L${b - 6} ${CY} L${b - 12} ${CY + 5}`, stroke: col, "stroke-opacity": 0.8, "stroke-width": 1.5, fill: "none", "stroke-linecap": "round", "stroke-linejoin": "round" } }, svg);
    });
    // one item per segment per beat; segment s starts its travel at offset OFF[s] within the beat
    const OFF = [0.0, 0.5, 1.0, 1.5], TRAVEL = 0.45;
    const items = SEG.map(([a, b, col], s) => {
      const el = h("div", { cls: "abs", style: { width: "14px", height: "14px", borderRadius: s === 3 ? "50%" : "4px", background: s === 0 ? C.paper : s === 1 ? C.schema : s === 2 ? C.synth : C.head, boxShadow: `0 0 12px ${rgba(s === 0 ? C.paper : s === 1 ? C.schema : s === 2 ? C.synth : C.head, 0.7)}` } }, L);
      return { el, a, b };
    });
    const RANKS = [
      [0.95, 0.9, 0.36, 0.12], [0.92, 0.81, 0.44, 0.2], [0.97, 0.7, 0.3, 0.16], [0.9, 0.86, 0.4, 0.1],
    ];

    U.push((t) => {
      t = mod(t, PERIOD);
      const beat = Math.floor(t / BEAT), tb = mod(t, BEAT);
      // items travel along their segment
      items.forEach((it, s) => {
        const k = clamp((tb - OFF[s]) / TRAVEL);
        const on = tb >= OFF[s] && tb <= OFF[s] + TRAVEL;
        tf(it.el, { x: lerp(it.a, it.b - 14, E.inOut(k)), y: CY - 7, o: on ? Math.sin(Math.PI * k) * 1.4 : 0 });
      });
      // node glows when an item arrives
      nodes.forEach((n, i) => {
        const arrive = OFF[i] + TRAVEL;
        const d = mod(tb - arrive, BEAT);
        const k = d < 0.9 ? Math.sin(Math.PI * clamp(d / 0.9)) : 0;
        n.c.style.boxShadow = `0 0 ${(26 * k).toFixed(1)}px ${rgba(n.nd.color, 0.35 * k)}, inset 0 1px 0 rgba(255,255,255,0.04)`;
      });
      // TabSchema: columns light in sequence across the period
      grid.forEach((g0) => {
        const ph = mod(t / BEAT * 2.5 - g0.k, 5);
        const kk = ph < 2.5 ? Math.sin(Math.PI * ph / 2.5) : 0;
        g0.cell.style.background = rgba(C.schema, 0.15 + 0.75 * kk);
      });
      // TabSynth: copies slide out of the real row each beat
      const ds = mod(tb - (OFF[1] + TRAVEL), BEAT);
      sy.forEach((row, r) => {
        if (r === 0) return;
        const k = E.brand(clamp((ds - (r - 1) * 0.12) / 0.5));
        const fade = 1 - E.inOut(clamp((ds - 1.5) / 0.4));
        row.style.transform = `translateY(${(-(r * 24) * (1 - k)).toFixed(2)}px)`;
        row.style.opacity = (k * fade).toFixed(3);
      });
      // TabHead: a snappy path per decision
      const dh = mod(tb - (OFF[2] + TRAVEL), BEAT);
      const paths = [[0, 2, 5], [0, 1, 4], [0, 2, 6], [0, 1, 3]];
      const hl = dh < 0.6 ? E.snap(clamp(dh / 0.12)) * (1 - clamp((dh - 0.4) / 0.2)) : 0;
      const pbeat = mod(beat - (dh > tb ? 1 : 0), PERIOD / BEAT);
      tree.update(1, 1, paths[pbeat], hl, 0);
      // the decision: bars snap to this beat's ranking when the item lands
      const dd = mod(tb - (OFF[3] + TRAVEL), BEAT);
      const cyc = mod(beat - ((tb - (OFF[3] + TRAVEL)) < 0 ? 1 : 0), PERIOD / BEAT);
      const kk = E.snap(clamp(dd / 0.22));
      bars.forEach((b, i) => {
        const v = RANKS[cyc][i] * kk;
        b.style.width = (132 * v).toFixed(1) + "px";
        b.style.background = i < 2 ? C.head : "rgba(230,233,242,0.45)";
      });
    });
    return { seek: (t) => { for (const u of U) u(t); } };
  }

  g.Loop = { build, W, H, PERIOD };
})(window);

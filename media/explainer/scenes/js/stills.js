// Stills: the video poster (1920x1080) and the Open Graph image (1200x630).
(function (g) {
  "use strict";
  const { h, place } = g.K;
  const { C, rgba, icon, Wordmark, Background, PHASES } = g.TA;

  function phaseCard(parent, i, x, y, w, hgt, big) {
    const ph = PHASES[i];
    const c = h("div", { cls: "abs card" }, parent);
    place(c, x, y, w, hgt);
    c.style.borderColor = rgba(ph.color, 0.45);
    h("div", { style: { position: "absolute", left: "22px", right: "22px", top: "0", height: "2px", background: ph.color } }, c);
    const s = big ? 1 : 0.78;
    h("div", { cls: "badge", text: String(i + 1), style: { position: "absolute", left: 24 * s + "px", top: 26 * s + "px", width: 30 * s + "px", height: 30 * s + "px", fontSize: 16 * s + "px", background: ph.color } }, c);
    h("div", { text: ph.name, style: { position: "absolute", left: 70 * s + "px", top: 22 * s + "px", fontWeight: 700, fontSize: 30 * s + "px", letterSpacing: "-0.015em", color: C.paper } }, c);
    h("div", { text: ph.sub, style: { position: "absolute", left: 70 * s + "px", top: 64 * s + "px", fontWeight: 600, fontSize: 19 * s + "px", color: ph.color, whiteSpace: "nowrap" } }, c);
    return c;
  }
  function arrow(parent, x, y, w, color) {
    const s = h("svg", { cls: "abs", attrs: { width: w, height: 20, viewBox: `0 0 ${w} 20` } }, parent);
    place(s, x, y - 10);
    h("path", { attrs: { d: `M2 10H${w - 4}M${w - 10} 4L${w - 4} 10L${w - 10} 16`, stroke: color, "stroke-width": 1.75, fill: "none", "stroke-linecap": "round", "stroke-linejoin": "round" } }, s);
  }
  const mult = (m) => `<span style="color:${C.head}"><span style="position:relative;top:-0.12em">~</span>${m}</span>`;

  function poster(stage, cfg) {
    const W = 1920, H = 1080;
    stage.style.width = W + "px"; stage.style.height = H + "px";
    const bg = Background(stage, W, H);
    bg.glow("schema", 500, 820, 900, 700, 0.6);
    bg.glow("synth", 960, 820, 900, 700, 0.55);
    bg.glow("head", 1420, 820, 900, 700, 0.6);
    bg.glow("paper", 700, 380, 1400, 800, 0.45);
    const L = h("div", { cls: "layer" }, stage);
    const wm = Wordmark(L, 52);
    wm.el.style.position = "absolute";
    place(wm.el, 120, 104);
    const hd = h("div", { cls: "abs", text: "Replace an LLM agent's closed-set decisions with a cheap classifier.", style: { fontWeight: 800, fontSize: "86px", lineHeight: 1.03, letterSpacing: "-0.025em", color: C.paper } }, L);
    place(hd, 120, 236, 1380);
    const sub = h("div", { cls: "abs", text: "Trained on the agent's own successful runs, with the surrounding agent left unchanged.", style: { fontWeight: 500, fontSize: "30px", color: C.text2, letterSpacing: "-0.01em", whiteSpace: "nowrap" } }, L);
    place(sub, 120, 470);
    const CW = 470, GAP = 135, Y = 640;
    for (let i = 0; i < 3; i++) {
      phaseCard(L, i, 120 + i * (CW + GAP), Y, CW, 118, true);
      if (i < 2) arrow(L, 120 + i * (CW + GAP) + CW + 22, Y + 59, GAP - 44, rgba(PHASES[i + 1].color, 0.8));
    }
    const stat = h("div", { cls: "abs", html: `${mult("10³×")} lower latency&nbsp;&nbsp;·&nbsp;&nbsp;${mult("10⁵×")} lower cost <span style="color:${C.dim};font-weight:500">per decision than the GPT-5.4 shortlister it replaces (AppWorld tool shortlisting)</span>`, style: { fontWeight: 700, fontSize: "30px", color: C.paper, whiteSpace: "nowrap", letterSpacing: "-0.01em" } }, L);
    place(stat, 120, 846);
    const foot = h("div", { cls: "abs", style: { display: "flex", alignItems: "center", gap: "14px", whiteSpace: "nowrap" } }, L);
    icon(foot, "arrowUR", 24, C.dim, 2);
    h("span", { text: cfg.repo_url, style: { fontWeight: 600, fontSize: "26px", color: C.text2 } }, foot);
    h("span", { text: "·  arXiv " + cfg.arxiv_id, style: { fontWeight: 500, fontSize: "24px", color: C.dim } }, foot);
    place(foot, 120, 950);
    return { seek() {}, duration: 0, size: [W, H] };
  }

  function og(stage, cfg) {
    const W = 1200, H = 630;
    stage.style.width = W + "px"; stage.style.height = H + "px";
    const bg = Background(stage, W, H);
    bg.glow("schema", 260, 420, 600, 480, 0.6);
    bg.glow("synth", 600, 420, 600, 480, 0.55);
    bg.glow("head", 940, 420, 600, 480, 0.6);
    const L = h("div", { cls: "layer" }, stage);
    const wm = Wordmark(L, 38);
    wm.el.style.position = "absolute";
    place(wm.el, 64, 54);
    const hd = h("div", { cls: "abs", text: "Distilling LLM Agent Components into Cheap Classifiers", style: { fontWeight: 800, fontSize: "56px", lineHeight: 1.06, letterSpacing: "-0.022em", color: C.paper } }, L);
    place(hd, 64, 146, 1000);
    const CW = 330, GAP = 39, Y = 324;
    for (let i = 0; i < 3; i++) {
      phaseCard(L, i, 64 + i * (CW + GAP), Y, CW, 96, false);
      if (i < 2) arrow(L, 64 + i * (CW + GAP) + CW + 6, Y + 48, GAP - 12, rgba(PHASES[i + 1].color, 0.8));
    }
    const stat = h("div", { cls: "abs", html: `${mult("10³×")} lower latency · ${mult("10⁵×")} lower cost <span style="color:${C.dim};font-weight:500">per decision vs the GPT-5.4 shortlister</span>`, style: { fontWeight: 700, fontSize: "25px", color: C.paper, whiteSpace: "nowrap" } }, L);
    place(stat, 64, 468);
    const foot = h("div", { cls: "abs", text: cfg.repo_url, style: { fontWeight: 600, fontSize: "21px", color: C.dim, whiteSpace: "nowrap" } }, L);
    place(foot, 64, 548);
    return { seek() {}, duration: 0, size: [W, H] };
  }

  // The README's video thumbnail: the poster with a play button and the running time. The button sits in the
  // open space right of the headline, so it covers no text.
  function thumb(stage, cfg) {
    const comp = poster(stage, cfg);
    const L = h("div", { cls: "layer" }, stage);
    // the pipeline card's border (1.5 px at 1200 wide); render.py cuts the corners at the same radius
    const frame = h("div", { cls: "abs", style: { border: "2.4px solid #232B45", borderRadius: "38px" } }, L);
    place(frame, 0, 0, 1920, 1080);
    const D = 156, X = 1650, Y = 330, PAD = 48, S = D + 2 * PAD, c = S / 2, r = D / 2;
    const s = h("svg", { cls: "abs", attrs: { width: S, height: S, viewBox: `0 0 ${S} ${S}` } }, L);
    place(s, X - c, Y - c);
    // a rounded triangle, its centroid on the circle's centre (optical centring)
    const tw = 0.3 * D, th = 0.36 * D, tx = c - tw / 3;
    s.innerHTML = `<defs>
      <linearGradient id="ring" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${C.schema}"/>
        <stop offset=".5" stop-color="${C.synth}"/><stop offset="1" stop-color="${C.head}"/></linearGradient>
      <radialGradient id="halo"><stop offset=".6" stop-color="${C.synth}" stop-opacity=".2"/>
        <stop offset="1" stop-color="${C.synth}" stop-opacity="0"/></radialGradient></defs>
      <circle cx="${c}" cy="${c}" r="${c}" fill="url(#halo)"/>
      <circle cx="${c}" cy="${c}" r="${r - 1.5}" fill="${rgba(C.ink2, 0.82)}" stroke="url(#ring)" stroke-width="3"/>
      <path d="M${tx} ${c - th / 2}L${tx + tw} ${c}L${tx} ${c + th / 2}Z" fill="${C.paper}" stroke="${C.paper}"
        stroke-width="9" stroke-linejoin="round"/>`;
    const sec = Math.round(cfg.explainer_seconds || 0);
    if (sec) {
      const chip = h("div", { cls: "abs", text: `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`, style: {
        fontFamily: '"JetBrains Mono"', fontWeight: 500, fontSize: "26px", lineHeight: "26px", color: C.paper,
        padding: "9px 14px", borderRadius: "12px", background: rgba(C.ink, 0.82), border: `1.5px solid ${C.line2}` } }, L);
      Object.assign(chip.style, { left: "auto", right: "120px", top: "942px" }); // level with the footer line
    }
    return comp;
  }

  g.Stills = { poster, og, thumb };
})(window);

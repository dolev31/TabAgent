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

  g.Stills = { poster, og };
})(window);

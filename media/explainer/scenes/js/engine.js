// Deterministic, time-driven animation helpers. Every frame is a pure function of t (seconds): there are no
// CSS transitions or animations, no timers and no requestAnimationFrame in the render path, so a headless
// browser can seek to any frame and screenshot it.
(function (g) {
  "use strict";

  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const lerp = (a, b, k) => a + (b - a) * k;

  // Cubic-bezier easing (the same solver as browsers use for CSS timing functions).
  function bezier(x1, y1, x2, y2) {
    const cx = 3 * x1, bx = 3 * (x2 - x1) - cx, ax = 1 - cx - bx;
    const cy = 3 * y1, by = 3 * (y2 - y1) - cy, ay = 1 - cy - by;
    const sx = (t) => ((ax * t + bx) * t + cx) * t;
    const sy = (t) => ((ay * t + by) * t + cy) * t;
    const dsx = (t) => (3 * ax * t + 2 * bx) * t + cx;
    function solve(x) {
      let t = x;
      for (let i = 0; i < 8; i++) {
        const e = sx(t) - x;
        if (Math.abs(e) < 1e-7) return t;
        const d = dsx(t);
        if (Math.abs(d) < 1e-7) break;
        t -= e / d;
      }
      let lo = 0, hi = 1;
      t = x;
      for (let i = 0; i < 60; i++) {
        const v = sx(t);
        if (Math.abs(v - x) < 1e-7) break;
        if (x > v) lo = t; else hi = t;
        t = (lo + hi) / 2;
      }
      return t;
    }
    return (x) => (x <= 0 ? 0 : x >= 1 ? 1 : sy(solve(x)));
  }

  const E = {
    brand: bezier(0.22, 1, 0.36, 1), // the brand's easeOutQuint
    inOut: bezier(0.65, 0, 0.35, 1),
    out: bezier(0.33, 1, 0.68, 1),
    in: bezier(0.55, 0, 0.75, 0.2),
    snap: bezier(0.16, 1, 0.3, 1), // the classifier: fast, decisive
    slow: bezier(0.42, 0, 0.58, 1), // the LLM: deliberate
    lin: (x) => x,
  };

  // progress of a beat that starts at a and lasts d seconds
  const P = (t, a, d, e = E.brand) => e(clamp((t - a) / d));
  // envelope: in over din from a, out over dout ending at b
  const ENV = (t, a, b, din = 0.6, dout = 0.45, ein = E.brand, eout = E.inOut) =>
    Math.min(P(t, a, din, ein), 1 - P(t, b - dout, dout, eout));

  const SVGNS = "http://www.w3.org/2000/svg";
  const SVG_TAGS = new Set([
    "svg", "g", "path", "rect", "circle", "line", "polyline", "polygon", "text", "tspan", "defs",
    "linearGradient", "radialGradient", "stop", "clipPath", "mask", "use", "ellipse", "pattern", "filter",
    "feGaussianBlur", "feMerge", "feMergeNode", "feDropShadow",
  ]);

  // h("div", {cls, text, html, style, attrs}, parent)
  function h(tag, o = {}, parent = null) {
    const el = SVG_TAGS.has(tag) ? document.createElementNS(SVGNS, tag) : document.createElement(tag);
    if (o.cls) el.setAttribute("class", o.cls);
    if (o.text != null) el.textContent = o.text;
    if (o.html != null) el.innerHTML = o.html;
    if (o.style) for (const k in o.style) el.style[k] = o.style[k];
    if (o.attrs) for (const k in o.attrs) el.setAttribute(k, o.attrs[k]);
    if (parent) parent.appendChild(el);
    return el;
  }

  // absolute placement (left/top/width/height in px)
  function place(el, x, y, w, hgt) {
    el.style.left = x + "px";
    el.style.top = y + "px";
    if (w != null) el.style.width = w + "px";
    if (hgt != null) el.style.height = hgt + "px";
    return el;
  }

  // transform + opacity; hidden elements are skipped by the painter
  function tf(el, o) {
    const x = o.x || 0, y = o.y || 0;
    const sx = o.sx != null ? o.sx : o.s != null ? o.s : 1;
    const sy = o.sy != null ? o.sy : o.s != null ? o.s : 1;
    const r = o.r || 0;
    let t = `translate(${x.toFixed(2)}px, ${y.toFixed(2)}px)`;
    if (sx !== 1 || sy !== 1) t += ` scale(${sx.toFixed(4)}, ${sy.toFixed(4)})`;
    if (r) t += ` rotate(${r.toFixed(2)}deg)`;
    el.style.transform = t;
    if (o.o != null) vis(el, o.o);
    return el;
  }
  function vis(el, o) {
    const v = clamp(o);
    el.style.opacity = v.toFixed(4);
    el.style.visibility = v <= 0.001 ? "hidden" : "visible";
  }
  function svgAttr(el, attrs) {
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  }

  // word-by-word reveal: returns the word spans
  function words(parent, text, cls = "w") {
    const out = [];
    const parts = String(text).split(/(\s+)/);
    for (const part of parts) {
      if (!part) continue;
      if (/^\s+$/.test(part)) parent.appendChild(document.createTextNode(" "));
      else out.push(h("span", { cls, html: part }, parent));
    }
    return out;
  }
  function revealWords(ws, t, a, opt = {}) {
    const stagger = opt.stagger != null ? opt.stagger : 0.028;
    const dur = opt.dur != null ? opt.dur : 0.55;
    const dy = opt.dy != null ? opt.dy : 18;
    ws.forEach((w, i) => {
      const k = P(t, a + i * stagger, dur);
      tf(w, { y: (1 - k) * dy, o: k });
    });
  }

  // deterministic pseudo-random
  function rng(seed) {
    let s = seed >>> 0 || 1;
    return () => {
      s ^= s << 13; s >>>= 0;
      s ^= s >> 17;
      s ^= s << 5; s >>>= 0;
      return s / 4294967296;
    };
  }

  // a smooth periodic pulse in [0, 1] (no hard edges: never a flash)
  const pulse = (t, period, phase = 0) => 0.5 - 0.5 * Math.cos(((t / period + phase) % 1) * 2 * Math.PI);

  g.K = { clamp, lerp, bezier, E, P, ENV, h, place, tf, vis, svgAttr, words, revealWords, rng, pulse, SVGNS };
})(window);

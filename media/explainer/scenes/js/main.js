// Entry point shared by every composition page. The renderer (render.py) waits for window.__ready, then
// calls window.__seek(t) and screenshots each frame. Opened in a browser, ?t=12.5 shows one frame and
// ?play=1 plays the composition in real time (a preview only; the renderer never uses it).
(function (g) {
  "use strict";
  async function json(url) {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(url + ": " + r.status);
    return r.json();
  }
  g.__boot = function (factory) {
    g.__ready = (async () => {
      const [script, cfg] = await Promise.all([json("../script.json"), json("../config.json")]);
      const over = g.__CONFIG_OVERRIDE || {};
      Object.assign(cfg, over);
      for (const c of script.cues) c.text = c.text.replace("{repo_url}", cfg.repo_url);
      await document.fonts.load('600 40px "Inter"');
      await document.fonts.load('800 40px "Inter"');
      await document.fonts.load('500 20px "JetBrains Mono"');
      await document.fonts.ready;
      const stage = document.getElementById("stage");
      const comp = factory(stage, script, cfg);
      g.__seek = (t) => comp.seek(t);
      g.__duration = comp.duration;
      g.__size = comp.size;
      const q = new URLSearchParams(location.search);
      if (q.get("play")) {
        const t0 = performance.now() / 1000 - Number(q.get("t") || 0);
        const loop = () => {
          let t = performance.now() / 1000 - t0;
          if (comp.duration) t = t % comp.duration;
          comp.seek(t);
          requestAnimationFrame(loop);
        };
        loop();
      } else {
        comp.seek(Number(q.get("t") || 0));
      }
      return { duration: comp.duration, size: comp.size };
    })();
  };
})(window);

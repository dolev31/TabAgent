"""Render the TabAgent explainer video and its companions from the HTML/SVG scenes in ``scenes/``.

Every frame is rendered deterministically: the page exposes ``__seek(t)``, a pure function of time, and a
headless system Chrome (driven by Playwright) screenshots each frame. ffmpeg (the self-contained binary
bundled with ``imageio-ffmpeg``) encodes them. Run from the repository root:

    uv run --with playwright --with imageio-ffmpeg --with pillow --with static-ffmpeg \\
        python media/explainer/render.py            # everything, into media/explainer/out/

Targets (default ``all``): ``explainer`` (MP4 + WebM), ``loop`` (MP4 + WebP), ``stills`` (poster, OG
image), ``captions`` (VTT + SRT), ``sheet`` (contact sheet), ``svg`` (docs/assets/pipeline-animated.svg),
``verify`` (ffprobe report and limits), ``frames --at 3.5,12`` (single PNGs for review).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import http.server
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
CACHE = ROOT / ".cache"
FONT_DIR = CACHE / "fonts"
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
CHROME_ARGS = [
    "--hide-scrollbars",
    "--force-color-profile=srgb",
    "--font-render-hinting=none",
    "--disable-lcd-text",
]

# Fonts are downloaded at render time from pinned upstream releases (SIL OFL 1.1) and never committed.
FONTS = {
    "InterVariable.woff2": (
        "https://raw.githubusercontent.com/rsms/inter/v4.1/docs/font-files/InterVariable.woff2",
        "693b77d4f32ee9b8bfc995589b5fad5e99adf2832738661f5402f9978429a8e3",
    ),
    "JetBrainsMono-wght.ttf": (
        "https://raw.githubusercontent.com/JetBrains/JetBrainsMono/v2.304/fonts/variable/JetBrainsMono%5Bwght%5D.ttf",
        "662a196d58f1183bf2d77428b6d5283fe3f45161ab021bea4036bc98e5cac016",
    ),
}

# compositions: page, size, duration (None = a still)
COMPS = {
    "explainer": ("scenes/explainer.html", (1920, 1080)),
    "loop": ("scenes/loop.html", (1280, 640)),
    "poster": ("scenes/stills.html?comp=poster", (1920, 1080)),
    "og": ("scenes/stills.html?comp=og", (1200, 630)),
}

LIMITS = {"mp4_mb": 15.0, "webp_mb": 1.5, "svg_kb": 60.0}


def log(*a):
    print("[render]", *a, flush=True)


def config(overrides: dict) -> dict:
    cfg = json.loads((ROOT / "config.json").read_text())
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


# ----------------------------------------------------------------------------------------------- fonts
def ensure_fonts() -> None:
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in FONTS.items():
        f = FONT_DIR / name
        if f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() == sha:
            continue
        log("downloading", name)
        data = urllib.request.urlopen(url, timeout=60).read()
        got = hashlib.sha256(data).hexdigest()
        if got != sha:
            sys.exit(f"{name}: SHA-256 mismatch ({got})")
        f.write_bytes(data)


# ----------------------------------------------------------------------------------------------- server
class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def serve() -> tuple[http.server.ThreadingHTTPServer, int]:
    handler = functools.partial(_Quiet, directory=str(ROOT))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


# ----------------------------------------------------------------------------------------------- frames
def _open(pw, port: int, comp: str, cfg: dict):
    page_path, (w, h) = COMPS[comp]
    browser = pw.chromium.launch(executable_path=CHROME, headless=True, args=CHROME_ARGS)
    page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
    page.add_init_script(f"window.__CONFIG_OVERRIDE = {json.dumps(cfg)};")
    page.goto(f"http://127.0.0.1:{port}/{page_path}")
    page.wait_for_function("window.__ready !== undefined")
    info = page.evaluate("window.__ready")
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    return browser, page, info, errors


def _worker(args):
    (
        port,
        comp,
        cfg,
        jobs,
    ) = args
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser, page, _, errors = _open(pw, port, comp, cfg)
        for t, path in jobs:
            page.evaluate(f"window.__seek({t!r})")
            page.screenshot(path=path, type="png")
            if errors:
                raise RuntimeError(f"page error at t={t}: {errors[0]}")
        browser.close()
    return len(jobs)


def render_frames(
    port: int, comp: str, cfg: dict, times: list[float], paths: list[Path], workers: int
) -> None:
    jobs = list(zip(times, [str(p) for p in paths], strict=True))
    if not jobs:
        return
    n = max(1, min(workers, len(jobs)))
    chunks = [jobs[i::n] for i in range(n)]  # interleaved, so every worker gets a similar mix of scenes
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(n) as pool:
        done = 0
        for k in pool.imap_unordered(_worker, [(port, comp, cfg, c) for c in chunks]):
            done += k
    log(f"{comp}: {done} frames in {time.time() - t0:.0f} s ({n} workers)")


def comp_duration(port: int, comp: str, cfg: dict) -> float:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser, _, info, _ = _open(pw, port, comp, cfg)
        browser.close()
    return float(info["duration"])


# ----------------------------------------------------------------------------------------------- ffmpeg
def ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def ffprobe() -> str | None:
    try:
        import static_ffmpeg.run

        _, probe = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
        return probe
    except Exception:  # noqa: BLE001 - the verify step falls back to ffmpeg -i
        return shutil.which("ffprobe")


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"command failed: {' '.join(cmd)}\n{r.stderr[-3000:]}")


BT709 = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]


def encode_mp4(frames: Path, fps: int, out: Path, crf: int, tune: str = "animation") -> None:
    run(
        [
            ffmpeg(),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frames / "%05d.png"),
            "-vf",
            "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryslow",
            "-tune",
            tune,
            "-crf",
            str(crf),
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
            *BT709,
            "-movflags",
            "+faststart",
            "-an",
            str(out),
        ]
    )


def encode_webm(frames: Path, fps: int, out: Path, crf: int) -> None:
    common = [
        ffmpeg(),
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frames / "%05d.png"),
        "-vf",
        "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-c:v",
        "libvpx-vp9",
        "-b:v",
        "0",
        "-crf",
        str(crf),
        "-row-mt",
        "1",
        "-tile-columns",
        "2",
        "-pix_fmt",
        "yuv420p",
        *BT709,
        "-an",
    ]
    passlog = str(frames.parent / "vp9pass")
    run(
        common
        + [
            "-pass",
            "1",
            "-passlogfile",
            passlog,
            "-deadline",
            "good",
            "-cpu-used",
            "4",
            "-f",
            "null",
            os.devnull,
        ]
    )
    run(common + ["-pass", "2", "-passlogfile", passlog, "-deadline", "good", "-cpu-used", "2", str(out)])
    for f in frames.parent.glob("vp9pass*"):
        f.unlink()


def encode_webp(frames: Path, fps: int, out: Path, quality: int, width: int | None = None) -> None:
    vf = f"fps={fps}" + (f",scale={width}:-1:flags=lanczos" if width else "")
    run(
        [
            ffmpeg(),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frames / "%05d.png"),
            "-vf",
            vf,
            "-c:v",
            "libwebp_anim",
            "-lossless",
            "0",
            "-quality",
            str(quality),
            "-compression_level",
            "6",
            "-preset",
            "drawing",
            "-loop",
            "0",
            "-an",
            str(out),
        ]
    )


# ----------------------------------------------------------------------------------------------- targets
def do_explainer(port, cfg, args):
    fps = args.fps or cfg["fps"]
    dur = comp_duration(port, "explainer", cfg)
    n = round(dur * fps)
    frames = OUT / "frames" / "explainer"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    times = [i / fps for i in range(n)]
    render_frames(port, "explainer", cfg, times, [frames / f"{i:05d}.png" for i in range(n)], args.workers)
    (frames / "fps.txt").write_text(str(fps))
    log("encoding MP4")
    encode_mp4(frames, fps, OUT / "tabagent-explainer.mp4", args.crf)
    log("encoding WebM (VP9, two passes)")
    encode_webm(frames, fps, OUT / "tabagent-explainer.webm", args.vp9_crf)


def do_loop(port, cfg, args):
    fps = cfg["loop_fps"]
    dur = comp_duration(port, "loop", cfg)
    n = round(dur * fps)
    frames = OUT / "frames" / "loop"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    times = [i / fps for i in range(n)]  # [0, dur): the frame at dur equals the frame at 0
    render_frames(port, "loop", cfg, times, [frames / f"{i:05d}.png" for i in range(n)], args.workers)
    encode_mp4(frames, fps, OUT / "tabagent-loop.mp4", 20)
    q = 82
    while True:
        encode_webp(frames, fps, OUT / "tabagent-loop.webp", q)
        mb = (OUT / "tabagent-loop.webp").stat().st_size / 1e6
        if mb <= LIMITS["webp_mb"] or q <= 50:
            break
        q -= 8
    log(f"loop WebP quality {q}: {mb:.2f} MB")


def do_stills(port, cfg, args):
    for comp, name in (("poster", "poster.png"), ("og", "og-video.png")):
        render_frames(port, comp, cfg, [0.0], [OUT / name], 1)


def fmt_ts(s: float, sep: str) -> str:
    ms = round(s * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"


def do_captions(cfg):
    script = json.loads((ROOT / "script.json").read_text())
    cues = [
        (c["start"], c["end"], c["text"].replace("{repo_url}", cfg["repo_url"]).replace("\n", " "))
        for c in script["cues"]
    ]
    vtt = ["WEBVTT", ""]
    srt = []
    for i, (a, b, text) in enumerate(cues, 1):
        vtt += [f"{fmt_ts(a, '.')} --> {fmt_ts(b, '.')}", text, ""]
        srt += [str(i), f"{fmt_ts(a, ',')} --> {fmt_ts(b, ',')}", text, ""]
    (OUT / "tabagent-explainer.vtt").write_text("\n".join(vtt), encoding="utf-8")
    (OUT / "tabagent-explainer.srt").write_text("\n".join(srt), encoding="utf-8")

    # accessibility: every line stays fully legible for at least 2.5 s (the narration reveals word by word,
    # 0.5 s per word staggered by 0.026 s, and fades out over the cue's last 0.32 s)
    def legible(a, b, text):
        return (b - 0.32) - (a + 0.5 + 0.026 * (len(text.split()) - 1))

    short = [(a, b, t) for a, b, t in cues if legible(a, b, t) < 2.5]
    if short:
        sys.exit(f"cues shorter than 2.5 s of legible time: {short}")
    log(f"captions: {len(cues)} cues")


def do_sheet(port, cfg, args):
    from PIL import Image, ImageDraw, ImageFont

    frames = OUT / "frames" / "explainer"
    fps = int((frames / "fps.txt").read_text()) if (frames / "fps.txt").exists() else None
    dur = comp_duration(port, "explainer", cfg)
    secs = list(range(int(dur) + 1))
    shots = []
    missing = []
    for s in secs:
        t = min(s + 0.5, dur - 1 / 60)  # mid-second: away from beat boundaries
        f = frames / f"{round(t * fps):05d}.png" if fps else None
        if f is None or not f.exists():
            f = OUT / "sheet" / f"t{s:03d}.png"
            missing.append((t, f))
        shots.append((s, t, f))
    if missing:
        (OUT / "sheet").mkdir(parents=True, exist_ok=True)
        render_frames(port, "explainer", cfg, [m[0] for m in missing], [m[1] for m in missing], args.workers)
    cols, tw, th, pad, lab = 8, 384, 216, 12, 26
    rows = (len(shots) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + pad + lab) + pad), (5, 8, 18))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(str(FONT_DIR / "JetBrainsMono-wght.ttf"), 16)
    except OSError:
        font = ImageFont.load_default()
    for i, (_s, t, f) in enumerate(shots):
        x = pad + (i % cols) * (tw + pad)
        y = pad + (i // cols) * (th + pad + lab)
        im = Image.open(f).convert("RGB").resize((tw, th), Image.LANCZOS)
        sheet.paste(im, (x, y + lab))
        d.text((x, y + 3), f"{t:5.1f} s", fill=(148, 163, 184), font=font)
    sheet.save(OUT / "contact-sheet.png")
    log(f"contact sheet: {len(shots)} frames")


def do_frames(port, cfg, args, comp="explainer"):
    times = [float(x) for x in args.at.split(",")]
    d = OUT / "review"
    d.mkdir(parents=True, exist_ok=True)
    render_frames(port, comp, cfg, times, [d / f"{comp}-{t:06.2f}.png" for t in times], args.workers)


def do_svg():
    sys.path.insert(0, str(ROOT))
    import make_svg

    make_svg.main()


def probe(path: Path) -> dict:
    exe = ffprobe()
    if exe:
        r = subprocess.run(
            [
                exe,
                "-v",
                "error",
                "-show_entries",
                "format=duration,size:stream=codec_name,width,height,r_frame_rate,pix_fmt,nb_frames",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        return json.loads(r.stdout or "{}")
    return {}


def do_verify():
    report = {}
    for name in (
        "tabagent-explainer.mp4",
        "tabagent-explainer.webm",
        "tabagent-loop.mp4",
        "tabagent-loop.webp",
    ):
        p = OUT / name
        if not p.exists():
            continue
        info = probe(p)
        st = (info.get("streams") or [{}])[0]
        fmt = info.get("format") or {}
        report[name] = {
            "MB": round(p.stat().st_size / 1e6, 2),
            "duration_s": round(float(fmt.get("duration", "nan")), 3) if fmt.get("duration") else None,
            "codec": st.get("codec_name"),
            "size": f"{st.get('width')}x{st.get('height')}",
            "fps": st.get("r_frame_rate"),
            "pix_fmt": st.get("pix_fmt"),
        }
    for name in (
        "poster.png",
        "og-video.png",
        "contact-sheet.png",
        "tabagent-explainer.vtt",
        "tabagent-explainer.srt",
    ):
        p = OUT / name
        if p.exists():
            report[name] = {"KB": round(p.stat().st_size / 1e3, 1)}
    svg = ROOT.parent.parent / "docs" / "assets" / "pipeline-animated.svg"
    if svg.exists():
        report["docs/assets/pipeline-animated.svg"] = {"KB": round(svg.stat().st_size / 1e3, 1)}
    print(json.dumps(report, indent=2))
    problems = []
    mp4 = report.get("tabagent-explainer.mp4")
    if mp4 and mp4["MB"] > LIMITS["mp4_mb"]:
        problems.append("explainer MP4 over 15 MB")
    if mp4 and mp4["pix_fmt"] != "yuv420p":
        problems.append("explainer MP4 is not yuv420p")
    webp = report.get("tabagent-loop.webp")
    if webp and webp["MB"] > LIMITS["webp_mb"]:
        problems.append("loop WebP over 1.5 MB")
    s = report.get("docs/assets/pipeline-animated.svg")
    if s and s["KB"] > LIMITS["svg_kb"]:
        problems.append("animated SVG over 60 KB")
    if problems:
        sys.exit("limits: " + "; ".join(problems))
    log("verify: all limits met")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("targets", nargs="*", default=["all"])
    ap.add_argument("--fps", type=int, help="explainer frame rate (default: config.json)")
    ap.add_argument("--workers", type=int, default=max(2, min(8, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--crf", type=int, default=18, help="x264 CRF of the explainer MP4")
    ap.add_argument("--vp9-crf", type=int, default=33, help="VP9 CRF of the explainer WebM")
    ap.add_argument("--repo-url", help="override config.json repo_url (the one place the URL lives)")
    ap.add_argument("--at", default="0", help="frames: comma-separated times in seconds")
    ap.add_argument("--comp", default="explainer", help="frames: composition (explainer, loop, poster, og)")
    args = ap.parse_args(argv)
    targets = set(args.targets)
    if "all" in targets:
        targets = {"explainer", "loop", "stills", "captions", "sheet", "svg", "verify"}
    cfg = config({"repo_url": args.repo_url})
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_fonts()
    srv, port = serve()
    try:
        if "frames" in targets:
            do_frames(port, cfg, args, args.comp)
        if "explainer" in targets:
            do_explainer(port, cfg, args)
        if "loop" in targets:
            do_loop(port, cfg, args)
        if "stills" in targets:
            do_stills(port, cfg, args)
        if "captions" in targets:
            do_captions(cfg)
        if "sheet" in targets:
            do_sheet(port, cfg, args)
        if "svg" in targets:
            do_svg()
        if "verify" in targets:
            do_verify()
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()

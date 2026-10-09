# TabAgent explainer video

A 73-second explainer of TabAgent (the problem, the idea, TabSchema, TabSynth, TabHead, the paper's headline
efficiency and accuracy, and how to start), with its companions: a seamless pipeline loop for the README hero,
a poster, an Open Graph image, captions and an animated SVG.

Everything is source: HTML, CSS and SVG scenes driven by a deterministic timeline, rendered frame by frame in
headless Chrome and encoded with ffmpeg. The renders go to `out/`, which git ignores; the final MP4 and WebM are
hosted outside the repository.

## Re-render

Requirements: Google Chrome (the renderer drives the installed browser; set `CHROME=/path/to/chrome` if it is
not at the macOS default) and [uv](https://docs.astral.sh/uv/). Python packages come from `uv run --with`, so
nothing is installed into the project. From the repository root:

```bash
uv run --no-project --with playwright --with imageio-ffmpeg --with pillow --with static-ffmpeg \
    --with fonttools --with brotli --with uharfbuzz \
    python media/explainer/render.py
```

This renders every target into `media/explainer/out/` and regenerates `docs/assets/pipeline-animated.svg`. A full
run takes about 12 minutes on a 10-core laptop, most of it the two-pass VP9 encode (the 4,380 frames take under
two minutes). Targets can be named one by one:

| Target | Output |
|---|---|
| `explainer` | `tabagent-explainer.mp4` (1920x1080, 60 fps, H.264 yuv420p, faststart) and `tabagent-explainer.webm` (VP9) |
| `loop` | `tabagent-loop.mp4` and `tabagent-loop.webp` (1280x640, 8 s, seamless) |
| `stills` | `poster.png` (1920x1080) and `og-video.png` (1200x630) |
| `captions` | `tabagent-explainer.vtt` and `tabagent-explainer.srt` |
| `sheet` | `contact-sheet.png`: one frame per second, for review |
| `svg` | `docs/assets/pipeline-animated.svg` (also `python media/explainer/make_svg.py`) |
| `verify` | an ffprobe report of the outputs and a check of the size limits |
| `frames --at 3.5,12` | single PNGs in `out/review/` (`--comp loop`, `poster` or `og` for the others) |

Options: `--fps 30` for a quicker explainer render, `--workers N` for the number of parallel Chrome instances,
`--crf` and `--vp9-crf` for the encoders' quality, `--repo-url` to override the repository URL for one render.

ffmpeg is the self-contained binary bundled with `imageio-ffmpeg` (a Homebrew ffmpeg that lost a library is
never used); ffprobe comes from `static-ffmpeg`. The two fonts, Inter and JetBrains Mono (SIL Open Font License
1.1), are downloaded on the first run from their pinned upstream releases, checked by SHA-256 and kept in
`.cache/fonts/`, which git ignores: they are not redistributed.

## Edit

- `config.json`: the constants shown on screen. `repo_url` is the one place the repository URL lives (the
  public name is not final); `install_cmd` and `onboard_cmd` are the two commands of the end card; `arxiv_id`.
- `script.json`: the narration, one cue per line of on-screen text, with its start and end in seconds. The
  captions are generated from it; `\n` in a cue breaks the line on screen only. The `captions` target refuses a
  cue that would stay fully legible for less than 2.5 s.
- `claims.md`: the source of every number and claim on screen. Update it with any change to the text.
- `scenes/explainer.html` and `scenes/js/explainer.js`: the seven scenes (scene windows in `S`). Every frame is
  a pure function of time: `__seek(t)` sets the whole picture, so there are no CSS transitions or timers. Avoid
  `will-change` on animated text: Chrome would reuse a raster cached mid-animation, and the frame would then
  depend on the frames rendered before it.
- `scenes/js/components.js`: the shared parts (mark, wordmark, rail, narration, LLM bubble, tree glyph, the
  feature-card table, code editor); `scenes/css/base.css`: the brand tokens.
- `scenes/loop.html`, `scenes/stills.html`: the loop and the stills; `make_svg.py`: the animated SVG.

To preview in a browser, serve this folder (`python -m http.server -d media/explainer`) and open
`scenes/explainer.html?play=1` (real time) or `scenes/explainer.html?t=31.5` (one frame).

## Design notes

- Brand spec: ink background, Inter for display and UI, JetBrains Mono for code and numbers, TabSchema cyan,
  TabSynth violet, TabHead amber, the LLM rose and used sparingly; the brand's easeOutQuint
  (`cubic-bezier(0.22, 1, 0.36, 1)`) for entrances, snappy beats for the classifier and slow ones for the LLM.
- Accessibility: text contrast is AA or better on the dark background (secondary text uses slate-400, 7.4:1,
  because the brand's slate is 4.0:1 on ink); no element flashes (the LLM's thinking dots pulse once every
  1.2 s); every line of text stays fully legible for at least 2.5 s; the video is silent and the captions carry
  the narration.
- The animated SVG has no script and no web font (its text is converted to paths), so it renders in a GitHub
  README `<img>`; under `prefers-reduced-motion: reduce` it shows a complete still diagram.

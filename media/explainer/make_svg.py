"""Generate docs/assets/pipeline-animated.svg: the TabSchema -> TabSynth -> TabHead pipeline as a pure SVG
with CSS animations (no script), for a GitHub README <img>. Its text is converted to glyph paths (Inter,
shaped with HarfBuzz), so the image needs no web font; under prefers-reduced-motion it is a still diagram.

    uv run --no-project --with fonttools --with brotli --with uharfbuzz python media/explainer/make_svg.py
"""

from __future__ import annotations

import io
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FONT = ROOT / ".cache" / "fonts" / "InterVariable.woff2"
OUT = ROOT.parent.parent / "docs" / "assets" / "pipeline-animated.svg"

INK, INK2, PAPER, DIM, LINE = "#0B1020", "#141A2E", "#F7F8FC", "#94A3B8", "#2A3350"
SCHEMA, SYNTH, HEAD, OK, GREY = "#22D3EE", "#A78BFA", "#F59E0B", "#34D399", "#8A93A8"
W, H, CY = 1200, 400, 200
PERIOD, BEAT, TRAVEL = 8, 2, 0.45  # seconds; a new item enters every BEAT and crosses a link in TRAVEL
OFF = [0.0, 0.5, 1.0, 1.5]  # when each link's item leaves, within the beat
SNAPPY, SMOOTH = "cubic-bezier(.16,1,.3,1)", "cubic-bezier(.65,0,.35,1)"


def el(tag: str, **attrs) -> str:
    """An empty SVG element; attribute names use _ for - (stroke_width -> stroke-width), class_ for class."""
    parts = []
    for k, v in attrs.items():
        name = "class" if k == "class_" else k.replace("_", "-")
        parts.append(f'{name}="{v}"')
    return f"<{tag} {' '.join(parts)}/>"


def pct(t: float, period: float = PERIOD) -> str:
    return f"{100 * t / period:.2f}%"


class Glyphs:
    """Shapes text with HarfBuzz and defines each distinct glyph once, as a path in font units."""

    def __init__(self):
        import uharfbuzz as hb
        from fontTools.ttLib import TTFont

        f = TTFont(str(FONT))
        f.flavor = None
        buf = io.BytesIO()
        f.save(buf)
        self.hb = hb
        self.face = hb.Face(hb.Blob(buf.getvalue()))
        self.upem = self.face.upem
        self.fonts = {}
        self.defs: dict[tuple[int, int], str] = {}

    def _font(self, weight: int, opsz: float):
        if (weight, opsz) not in self.fonts:
            ft = self.hb.Font(self.face)
            ft.set_variations({"wght": weight, "opsz": opsz})
            self.fonts[(weight, opsz)] = ft
        return self.fonts[(weight, opsz)]

    def run(
        self, text: str, x: float, y: float, size: float, weight: int, fill: str, anchor: str = "start"
    ) -> str:
        """A group drawing text with its baseline at (x, y)."""
        from fontTools.pens.svgPathPen import SVGPathPen

        ft = self._font(weight, max(14, min(32, size)))
        buf = self.hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        self.hb.shape(ft, buf, {"kern": True, "liga": True})
        uses, pen_x = [], 0
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions, strict=True):
            key = (weight, info.codepoint)
            if key not in self.defs:
                pen = SVGPathPen(None, ntos=lambda v: str(round(v)))
                ft.draw_glyph_with_pen(info.codepoint, pen)
                self.defs[key] = pen.getCommands()
            if self.defs[key]:
                uses.append(f'<use href="#g{weight}_{info.codepoint}" x="{pen_x + pos.x_offset}"/>')
            pen_x += pos.x_advance
        s = size / self.upem
        width = pen_x * s
        x0 = {"start": x, "middle": x - width / 2, "end": x - width}[anchor]
        matrix = f"matrix({s:.5f} 0 0 {-s:.5f} {x0:.1f} {y:.1f})"
        return f'<g fill="{fill}" transform="{matrix}">{"".join(uses)}</g>'

    def defs_svg(self) -> str:
        return "".join(f'<path id="g{w}_{g}" d="{d}"/>' for (w, g), d in self.defs.items() if d)


def check(cx: float, cy: float) -> str:
    return el("circle", cx=cx, cy=cy, r=8, fill=OK) + el(
        "path",
        d=f"M{cx - 4} {cy}l2.8 2.8 5-5",
        fill="none",
        stroke=INK,
        stroke_width=2,
        stroke_linecap="round",
        stroke_linejoin="round",
    )


def main():
    G = Glyphs()
    body: list[str] = []
    css = [
        "@keyframes pulse{0%,100%{stroke-opacity:.45}12%{stroke-opacity:1}30%{stroke-opacity:.45}}",
        "@keyframes cell{0%,100%{fill-opacity:.16}25%{fill-opacity:.95}50%{fill-opacity:.16}}",
        f".nd{{animation:pulse {BEAT}s infinite}}.cl{{animation:cell {BEAT}s infinite}}",
    ]

    # frame, dot grid and the phase glows
    body += [
        el("rect", width=W, height=H, rx=24, fill=INK),
        el(
            "rect",
            x=0.75,
            y=0.75,
            width=W - 1.5,
            height=H - 1.5,
            rx=23.25,
            fill="none",
            stroke="#232B45",
            stroke_width=1.5,
        ),
        el("rect", width=W, height=H, rx=24, fill="url(#dots)"),
    ]
    for cx, c in ((311, SCHEMA), (581, SYNTH), (851, HEAD)):
        body.append(el("ellipse", cx=cx, cy=CY, rx=230, ry=170, fill=f"url(#gl{c[1:]})"))

    # the agent's logs: three stacked successful runs
    for i in (2, 1, 0):
        x, y = 40 + i * 8, CY - 76 + i * 8
        body.append(
            el("rect", x=x, y=y, width=96, height=128, rx=12, fill=INK2, stroke=LINE, stroke_width=1.5)
        )
    x, y = 40, CY - 76
    body.append(check(x + 21, y + 22))
    for j, w in enumerate((62, 50, 66, 44)):
        body.append(
            el(
                "rect",
                x=x + 13,
                y=y + 44 + j * 20,
                width=w,
                height=7,
                rx=3.5,
                fill=PAPER if j == 3 else "#3A4560",
            )
        )
    body.append(G.run("Agent logs", 40, CY + 98, 15, 600, PAPER))
    body.append(G.run("successful runs", 40, CY + 118, 12, 600, DIM))

    # the three phases
    phases = [
        (196, SCHEMA, "1", "TabSchema", "discovers features"),
        (466, SYNTH, "2", "TabSynth", "augments the rows"),
        (736, HEAD, "3", "TabHead", "decides in milliseconds"),
    ]
    NW, NY, NH = 230, 100, 200
    for i, (x, c, n, title, sub) in enumerate(phases):
        body += [
            el(
                "rect",
                class_="nd",
                style=f"animation-delay:{OFF[i] + TRAVEL - BEAT:.2f}s",
                x=x,
                y=NY,
                width=NW,
                height=NH,
                rx=14,
                fill=INK2,
                stroke=c,
                stroke_opacity=0.45,
                stroke_width=1.5,
            ),
            el("rect", x=x + 18, y=NY, width=NW - 36, height=2, fill=c),
            el("circle", cx=x + 29, cy=NY + 31, r=11, fill=c),
            G.run(n, x + 29, NY + 35.5, 12.5, 700, INK, anchor="middle"),
            G.run(title, x + 18, NY + 80, 23, 700, PAPER),
            G.run(sub, x + 18, NY + 104, 13.5, 600, c),
        ]

    # TabSchema: a feature grid whose columns light in a sweep
    x = 196
    for k in range(5):
        for r in range(3):
            body.append(
                el(
                    "rect",
                    class_="cl",
                    style=f"animation-delay:{k * 0.25 - 2:.2f}s",
                    x=x + 18 + k * 40,
                    y=NY + 128 + r * 20,
                    width=34,
                    height=12,
                    rx=3,
                    fill=SCHEMA,
                    fill_opacity=0.95 if k in (1, 2) else 0.16,
                )
            )

    # TabSynth: a real row and its two paraphrased copies sliding out of it
    x = 466
    body.append(el("rect", x=x + 18, y=NY + 128, width=194, height=12, rx=3, fill=GREY))
    # keyframes run in local time (0 = the item's arrival); a negative delay sets the phase
    delay = OFF[1] + TRAVEL - BEAT
    for r in (1, 2):
        a = (r - 1) * 0.12
        css.append(
            f"@keyframes cp{r}{{0%,{pct(a, BEAT)}{{transform:translateY({-20 * r}px);opacity:0}}"
            f"{pct(a + 0.5, BEAT)},{pct(1.5, BEAT)}{{transform:none;opacity:1}}"
            f"{pct(1.9, BEAT)},100%{{opacity:0}}}}"
            f".cp{r}{{animation:cp{r} {BEAT}s {delay:.2f}s infinite}}"
        )
        body.append(
            el("rect", class_=f"cp{r}", x=x + 18, y=NY + 128 + r * 20, width=194, height=12, rx=3, fill=SYNTH)
        )

    # TabHead: the tree, lighting one decision path per beat (four decisions per period)
    nodes = [(100, 8), (40, 86), (160, 86), (10, 164), (70, 164), (130, 164), (190, 164)]
    edges = [
        "M120 36V61H60V86",
        "M120 36V61H180V86",
        "M60 114V139H30V164",
        "M60 114V139H90V164",
        "M180 114V139H150V164",
        "M180 114V139H210V164",
    ]
    paths = [[0, 2, 5], [0, 1, 4], [0, 2, 6], [0, 1, 3]]
    tree = ['<g transform="translate(815 224) scale(.3)">']
    tree += [
        el("path", d=d, fill="none", stroke=HEAD, stroke_opacity=0.6, stroke_width=5, stroke_linejoin="round")
        for d in edges
    ]
    delay = OFF[2] + TRAVEL - PERIOD  # local time 0 = the first decision's arrival
    for i, (nx, ny) in enumerate(nodes):
        frames = ["0%{fill-opacity:.12}"]
        for j, p in enumerate(paths):
            if i in p:
                a = j * BEAT
                frames.append(
                    f"{pct(a)}{{fill-opacity:.12}}{pct(a + 0.08)},{pct(a + 0.4)}{{fill-opacity:1}}"
                    f"{pct(a + 0.6)}{{fill-opacity:.12}}"
                )
        frames.append("100%{fill-opacity:.12}")
        css.append(
            f"@keyframes tn{i}{{{''.join(frames)}}}.tn{i}{{animation:tn{i} {PERIOD}s {delay:.2f}s infinite}}"
        )
        tree.append(
            el(
                "rect",
                class_=f"tn{i}",
                x=nx,
                y=ny,
                width=40,
                height=28,
                rx=8,
                fill=HEAD,
                fill_opacity=1 if i in paths[0] else 0.12,
                stroke=HEAD,
                stroke_width=5,
            )
        )
    tree.append("</g>")
    body.append("".join(tree))

    # the decision: ranked bars that snap to each new decision's scores
    dx, dy = 1010, CY - 66
    body.append(
        el("rect", x=dx, y=dy, width=150, height=132, rx=12, fill=INK2, stroke=LINE, stroke_width=1.5)
    )
    ranks = [[0.95, 0.9, 0.36, 0.12], [0.92, 0.81, 0.44, 0.2], [0.97, 0.7, 0.3, 0.16], [0.9, 0.86, 0.4, 0.1]]
    delay = OFF[3] + TRAVEL - PERIOD  # local time 0 = the first decision's landing
    for b in range(4):
        y = dy + 26 + b * 24
        body.append(el("rect", x=dx + 16, y=y, width=118, height=8, rx=4, fill="#E6E9F2", fill_opacity=0.08))
        frames = [f"0%{{transform:scaleX({ranks[3][b]})}}"]  # the last decision of the period, until this one
        for j in range(4):
            a, prev = j * BEAT, ranks[(j - 1) % 4][b]
            frames.append(
                f"{pct(a)}{{transform:scaleX({prev})}}{pct(a + 0.02)}{{transform:scaleX(0)}}"
                f"{pct(a + 0.24)}{{transform:scaleX({ranks[j][b]})}}"
            )
        frames.append(f"100%{{transform:scaleX({ranks[3][b]})}}")
        css.append(
            f"@keyframes rb{b}{{{''.join(frames)}}}"
            f".rb{b}{{animation:rb{b} {PERIOD}s {delay:.2f}s infinite {SNAPPY}}}"
        )
        body.append(
            el(
                "rect",
                class_=f"rb{b} bar",
                x=dx + 16,
                y=y,
                width=118,
                height=8,
                rx=4,
                fill=HEAD if b < 2 else GREY,
                style=f"transform:scaleX({ranks[0][b]})",
            )
        )
    body.append(G.run("Ranked shortlist", dx, CY + 98, 15, 600, PAPER))
    body.append(G.run("one classifier pass", dx, CY + 118, 12, 600, DIM))

    # links, each with an item crossing it once per beat
    links = [(152, 196, "#CBD5E1"), (426, 466, SCHEMA), (696, 736, SYNTH), (966, 1010, HEAD)]
    for s, (a, b, c) in enumerate(links):
        body.append(
            el(
                "path",
                d=f"M{a + 6} {CY}H{b - 7}M{b - 13} {CY - 5}l6 5-6 5",
                fill="none",
                stroke=c,
                stroke_opacity=0.7,
                stroke_width=1.5,
                stroke_linecap="round",
                stroke_linejoin="round",
            )
        )
        p0, p1 = OFF[s], OFF[s] + TRAVEL
        css.append(
            f"@keyframes mv{s}{{0%,{pct(p0, BEAT)}{{transform:translateX(0);opacity:0}}"
            f"{pct(p0 + 0.06, BEAT)},{pct(p1 - 0.06, BEAT)}{{opacity:1}}"
            f"{pct(p1, BEAT)},100%{{transform:translateX({b - a - 22}px);opacity:0}}}}"
            f".mv{s}{{animation:mv{s} {BEAT}s infinite {SMOOTH}}}"
        )
        body.append(el("circle", class_=f"mv{s} it", cx=a + 10, cy=CY, r=5, fill=c))

    css.append(
        ".bar{transform-box:fill-box;transform-origin:left center}.it{opacity:0}"
        "@media (prefers-reduced-motion:reduce){*{animation:none!important}}"
    )
    dots = '<circle cx="12" cy="12" r="1" fill="#E6E9F2" fill-opacity=".06"/>'
    defs = [f'<pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse">{dots}</pattern>']
    for c in (SCHEMA, SYNTH, HEAD):
        defs.append(
            f'<radialGradient id="gl{c[1:]}"><stop offset="0" stop-color="{c}" stop-opacity=".16"/>'
            f'<stop offset="1" stop-color="{c}" stop-opacity="0"/></radialGradient>'
        )
    defs.append(G.defs_svg())
    title = (
        "How TabAgent works: an agent's successful runs (its logs) go through TabSchema, which discovers "
        "features, TabSynth, which augments the rows, and TabHead, a classifier that decides in milliseconds "
        "and returns a ranked shortlist in one pass."
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        f'aria-label="{title}"><title>{title}</title><style>{"".join(css)}</style>'
        f"<defs>{''.join(defs)}</defs>{''.join(body)}</svg>\n"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg, encoding="utf-8")
    print(f"[svg] {OUT.relative_to(ROOT.parent.parent)}: {len(svg.encode()) / 1000:.1f} KB")


if __name__ == "__main__":
    main()

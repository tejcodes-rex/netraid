"""
Build the submission PDFs that the brief requires (.pdf deliverables):
  dist/NetraID-Pitch.pdf            from deck/index.html
  dist/NetraID-Technical-Docs.pdf   from README + docs/*.md combined

Run: python tools/build_pdfs.py
Needs: playwright (chromium installed), markdown.
"""
from __future__ import annotations
import os
import re
import subprocess
import tempfile
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DIST.mkdir(exist_ok=True)

DOC_ORDER = [
    ("README.md", ROOT / "README.md"),
    ("Architecture", ROOT / "docs/ARCHITECTURE.md"),
    ("Model Pipeline", ROOT / "docs/MODEL_PIPELINE.md"),
    ("Liveness & Anti-Spoofing", ROOT / "docs/LIVENESS.md"),
    ("Benchmarks", ROOT / "docs/BENCHMARKS.md"),
    ("Integration Guide", ROOT / "docs/INTEGRATION.md"),
    ("Calibrating the Liveness Gates", ROOT / "docs/CALIBRATION.md"),
    ("Security & Privacy", ROOT / "docs/SECURITY_PRIVACY.md"),
    ("Build from Source", ROOT / "docs/BUILD.md"),
    ("Requirement Compliance", ROOT / "docs/COMPLIANCE.md"),
]

DOC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800;900&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
@page{size:A4;margin:18mm 16mm}
*{box-sizing:border-box}
body{font-family:'IBM Plex Sans',sans-serif;color:#16201C;font-size:11.5px;line-height:1.55}
h1,h2,h3{font-family:'Archivo',sans-serif;color:#0B3D2A;letter-spacing:-0.01em;line-height:1.15}
h1{font-size:24px;border-bottom:3px solid #1FD27A;padding-bottom:6px;margin:26px 0 12px;break-before:page}
h1:first-of-type{break-before:auto}
h2{font-size:17px;margin:20px 0 8px}
h3{font-size:13.5px;margin:14px 0 6px;color:#0C7C49}
a{color:#0C7C49;text-decoration:none}
.diagram{margin:14px 0;text-align:center;break-inside:avoid}
.diagram svg{max-width:100%;height:auto}
code{font-family:'IBM Plex Mono',monospace;background:#EAF3EE;padding:1px 5px;border-radius:4px;font-size:10.5px}
pre{background:#0E1714;color:#E9F1ED;padding:12px 14px;border-radius:8px;font-size:10px;line-height:1.45;white-space:pre-wrap;word-break:break-word;break-inside:avoid}
tr{break-inside:avoid}
td,th{word-break:break-word;overflow-wrap:anywhere}
table{table-layout:fixed}
pre code{background:none;color:inherit;padding:0}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:10.5px}
th,td{border:1px solid #CBD8D1;padding:6px 9px;text-align:left;vertical-align:top}
th{background:#EAF3EE;font-family:'IBM Plex Mono',monospace;font-size:9.5px;text-transform:uppercase;letter-spacing:0.06em}
blockquote{border-left:3px solid #1FD27A;margin:10px 0;padding:4px 14px;color:#42554C;background:#F4F9F6}
.cover{height:240mm;display:flex;flex-direction:column;justify-content:center;break-after:page}
.cover .logo{width:64px;height:64px;border-radius:16px;background:radial-gradient(circle at 30% 25%,#27e98a,#0C7C49);margin-bottom:24px}
.cover h1{font-size:46px;border:none;color:#0B3D2A;margin:0}
.cover .tag{font-family:'IBM Plex Mono',monospace;color:#0C7C49;letter-spacing:0.2em;text-transform:uppercase;margin-top:10px;font-size:12px}
.cover .meta{margin-top:28px;font-family:'IBM Plex Mono',monospace;color:#42554C;font-size:12px;line-height:2}
"""

COVER = """
<div class="cover">
  <div class="logo"></div>
  <h1>NetraID</h1>
  <div class="tag">Offline Edge Face Authentication</div>
  <div class="meta">
    NHAI Innovation Hackathon 7.0 · Datalake 3.0<br>
    Technical Documentation<br>
    Model architecture · Integration · Performance benchmarks
  </div>
</div>
"""


MERMAID_RE = re.compile(r"```mermaid\n(.*?)```", re.S)


def render_mermaid(text: str) -> str:
    """Replace ```mermaid fences with inline SVG.

    GitHub renders these natively; a PDF does not, and shipping a technical
    document whose architecture diagram is a wall of `flowchart LR` source is
    worse than shipping no diagram. Rendered here with the same mermaid CLI, so
    the PDF and the repository cannot drift apart.
    """
    def one(m: "re.Match[str]") -> str:
        src = m.group(1)
        with tempfile.TemporaryDirectory() as d:
            mmd = Path(d) / "d.mmd"
            svg = Path(d) / "d.svg"
            mmd.write_text(src, encoding="utf-8")
            r = subprocess.run(
                ["npx", "-y", "@mermaid-js/mermaid-cli", "-i", str(mmd), "-o", str(svg),
                 "-b", "white"],
                capture_output=True, text=True, shell=(os.name == "nt"),
            )
            if r.returncode != 0 or not svg.exists():
                print(f"  mermaid render failed, keeping source: {r.stderr.strip()[:120]}")
                return m.group(0)
            body = svg.read_text(encoding="utf-8")
            body = body[body.index("<svg"):]
            return f'<div class="diagram">{body}</div>'
    return MERMAID_RE.sub(one, text)


def build_docs_pdf():
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"])
    parts = [COVER]
    for title, path in DOC_ORDER:
        if not path.exists():
            continue
        md.reset()
        text = path.read_text(encoding="utf-8")
        if "```mermaid" in text:
            print(f"  rendering diagrams in {path.name}")
            text = render_mermaid(text)
        parts.append(md.convert(text))
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{DOC_CSS}</style></head><body>{''.join(parts)}</body></html>"
    tmp = DIST / "_docs.html"
    tmp.write_text(html, encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(tmp.resolve().as_uri(), wait_until="networkidle")
        pg.wait_for_timeout(1200)
        pg.pdf(path=str(DIST / "NetraID-Technical-Docs.pdf"),
               format="A4", print_background=True,
               margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        b.close()
    tmp.unlink(missing_ok=True)
    print("wrote dist/NetraID-Technical-Docs.pdf")


def build_deck_pdf():
    deck = (ROOT / "deck" / "index.html").resolve().as_uri()
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 720})
        pg.goto(deck, wait_until="networkidle")
        pg.wait_for_timeout(1500)  # let web fonts + JS render
        pg.emulate_media(media="print")
        pg.pdf(path=str(DIST / "NetraID-Pitch.pdf"),
               width="1280px", height="720px", print_background=True,
               margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        b.close()
    print("wrote dist/NetraID-Pitch.pdf")


if __name__ == "__main__":
    build_deck_pdf()
    build_docs_pdf()

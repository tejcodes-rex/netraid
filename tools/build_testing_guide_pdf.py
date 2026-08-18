"""Build the standalone testing-guide PDF from docs/TESTING_GUIDE.md.

Same house style as the other submission PDFs, but its own short document, so an
evaluator can pick up the APK and the guide without opening a 33-page technical
manual.
"""
import io
import sys
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DIST.mkdir(exist_ok=True)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
@page{size:A4;margin:18mm 16mm}
*{box-sizing:border-box}
body{font-family:'IBM Plex Sans',sans-serif;color:#16201C;font-size:11.5px;line-height:1.6}
h1{font-family:'Archivo',sans-serif;font-size:23px;color:#0B3D2A;border-bottom:3px solid #1FD27A;
   padding-bottom:7px;margin:0 0 4px;letter-spacing:-0.01em}
h2{font-family:'Archivo',sans-serif;font-size:15.5px;color:#0B3D2A;margin:22px 0 7px;
   padding-top:10px;border-top:1px solid #E2ECE6;break-after:avoid}
h3{font-family:'Archivo',sans-serif;font-size:12.5px;color:#0C7C49;margin:15px 0 5px;break-after:avoid}
p{margin:0 0 9px}
strong{color:#0B3D2A}
code{font-family:'IBM Plex Mono',monospace;background:#EAF3EE;padding:1px 5px;border-radius:4px;
     font-size:10.5px;color:#0B3D2A}
table{width:100%;border-collapse:collapse;margin:10px 0 14px;font-size:10.8px;break-inside:avoid}
th{background:#EAF3EE;text-align:left;padding:7px 9px;border:1px solid #D4E4DB;
   font-family:'Archivo',sans-serif;font-size:9.5px;letter-spacing:.05em;text-transform:uppercase;
   color:#0B3D2A}
td{padding:7px 9px;border:1px solid #D4E4DB;vertical-align:top}
hr{border:0;border-top:1px solid #E2ECE6;margin:18px 0}
.cover{margin-bottom:16px}
.cover .sub{color:#4A5F55;font-size:12px;margin:2px 0 0}
.cover .meta{color:#6B7F75;font-size:10.5px;margin-top:8px;font-family:'IBM Plex Mono',monospace}
"""

COVER = """
<div class="cover">
  <h1>NetraID, Trying the App</h1>
  <p class="sub">Offline facial recognition and liveness detection for NHAI Datalake 3.0</p>
  <p class="meta">Evaluation guide for Android. No network connection required.</p>
</div>
"""

src = (ROOT / "docs" / "TESTING_GUIDE.md").read_text(encoding="utf-8")
# The cover carries the title, so drop the document's own H1 and its lead-in rule.
lines = src.split("\n")
if lines and lines[0].startswith("# "):
    lines = lines[1:]
src = "\n".join(lines).lstrip("\n")

md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists"])
html = ("<!doctype html><html><head><meta charset='utf-8'><style>" + CSS +
        "</style></head><body>" + COVER + md.convert(src) + "</body></html>")

tmp = DIST / "_guide.html"
tmp.write_text(html, encoding="utf-8")

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto(tmp.resolve().as_uri(), wait_until="networkidle")
    pg.wait_for_timeout(900)
    pg.pdf(path=str(DIST / "NetraID-Testing-Guide.pdf"), format="A4", print_background=True,
           margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
    b.close()

tmp.unlink(missing_ok=True)
print("wrote dist/NetraID-Testing-Guide.pdf")

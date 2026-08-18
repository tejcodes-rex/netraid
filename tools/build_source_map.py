"""Regenerate the dependency graph inside docs/index.html.

The graph is derived from the actual import statements in app/src, rendered
with the mermaid CLI, and inlined as SVG so the page stays self-contained and
opens offline. Run this after adding or removing a module, so the map cannot
drift away from the code it describes.

Needs npx (for @mermaid-js/mermaid-cli), and docs/index.html WITHOUT an
existing dependency-graph section: check that file out from git first.
"""
import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.environ.get("NETRAID_BUILD_DIR", os.path.join(ROOT, "build"))
PAGE = os.path.join(ROOT, "docs", "index.html")

svg = io.open(os.path.join(SP, "mmd", "deps.svg"), encoding="utf-8").read()
svg = svg[svg.index("<svg"):]
# Pin the SVG to its intrinsic viewBox width. Mermaid emits width="100%" with a
# max-width style, which collapses the graph to whatever the column happens to be
# and makes 22 labels unreadable. The wrapper scrolls horizontally instead.
_vb = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', svg)
_w = int(float(_vb.group(1))) if _vb else 2400
svg = re.sub(r'\sstyle="[^"]*max-width[^"]*"', ' style="background-color:transparent"', svg, count=1)
svg = svg.replace('width="100%"', 'width="%d"' % _w, 1)
# Rename EVERY occurrence, not just the root element: mermaid scopes its own
# style rules with "#my-svg ...", so renaming only the element left every fill
# rule pointing at an id that no longer existed and the graph rendered black.
svg = svg.replace("my-svg", "depsvg")

LABELS = {
    "api": "index.ts", "fproc": "frameProcessor.ts", "live": "liveness.ts", "chroma": "chroma.ts",
    "types": "types.ts", "recog": "recognition.ts", "align": "align.ts", "blaze": "blazeface.ts",
    "qual": "quality.ts", "math": "math.ts", "store": "store.ts", "keys": "keys.ts",
    "sync": "sync.ts", "calib": "calibration.ts", "lamp": "screenLamp.ts", "imaging": "imaging.ts",
    "demopipe": "demoPipeline.ts", "demoassets": "demoAssets.ts", "home": "HomeScreen.tsx",
    "enroll": "EnrollScreen.tsx", "verify": "VerifyScreen.tsx", "demoscr": "PipelineDemoScreen.tsx",
}
ROLE = {
    "api": "Public API. The only surface a host application calls.",
    "fproc": "The real-time worklet: detection, landmarks, the challenge state machine, continuity binding and the passive anti-spoof read.",
    "live": "Challenge state machine and the combined liveness verdict.",
    "chroma": "Screen-illumination challenge. Off by default.",
    "types": "All types and DEFAULT_CONFIG. Every threshold lives here.",
    "recog": "MobileFaceNet embedding through TFLite.",
    "align": "5-point similarity transform to the 112x112 crop the embedder expects.",
    "blaze": "BlazeFace anchor decoding and non-maximum suppression.",
    "qual": "Sharpness and exposure gates on the aligned crop.",
    "math": "Cosine similarity, robust averaging, median and consensus.",
    "store": "op-sqlite with SQLCipher: embeddings and the attendance queue.",
    "keys": "Database key held in the Android Keystore or the iOS Keychain.",
    "sync": "Offline-first queue flush, then purge on server acknowledgement.",
    "calib": "One machine-readable line per attempt, for calibration.",
    "lamp": "Android-only screen brightness bridge, behind a platform guard.",
    "imaging": "Pure raster helpers for the camera-free path.",
    "demopipe": "Runs the full pipeline over bundled frames, with no camera.",
    "demoassets": "Generated frame data for the demo path.",
    "home": "Dashboard screen.",
    "enroll": "Enrollment screen, six-shot capture.",
    "verify": "Verification screen: challenge flow and verdict.",
    "demoscr": "Pipeline demo screen.",
}

SECTION = """<section id="deps">
  <h2>Dependency graph</h2>
  <p class="sub">Generated from the actual <code>import</code> statements in the shipped source rather than drawn by
  hand: 22 modules, 47 edges. <strong>Click any module</strong> to isolate it and see exactly what it imports and what
  imports it.</p>
  <div class="dep-stage" id="dep-stage">
    <div class="dep-tools">
      <button id="dep-reset" class="btn">Show everything</button>
      <button id="dep-full" class="btn">Full screen</button>
      <span id="dep-status" class="dim-note">Nothing selected</span>
    </div>
    <div class="dep-wrap">__SVG__</div>
  </div>
  <div id="dep-detail" class="dep-detail"></div>
  <div class="note">
    <strong>How to read it.</strong> An arrow from A to B means A imports B, so B knows nothing about A.
    <code>types.ts</code> sits at the bottom because everything depends on the configuration and nothing depends back
    on it. <code>index.ts</code> is the only module a host application calls. The screens depend on the core and the
    core never depends on a screen, which is what makes this a drop-in module rather than an app.
  </div>
</section>

"""

CSS = """  .dep-wrap{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:16px;
    overflow-x:auto;margin:10px 0 14px}
  /* Natural size, so the labels stay readable; the wrapper scrolls instead. */
  .dep-wrap svg{max-width:none !important;width:auto;height:auto;display:block;margin:0 auto}
  .dep-tools{display:flex;align-items:center;gap:14px;margin:6px 0 4px;flex-wrap:wrap}
  .btn{background:var(--panel2);border:1px solid var(--line);color:var(--ink);border-radius:6px;
    padding:6px 13px;font:inherit;font-size:13px;cursor:pointer}
  .btn:hover{border-color:var(--accent);color:var(--accent)}
  .dim-note{color:var(--dim);font-size:13px}
  .dep-stage:fullscreen{background:var(--bg);padding:18px 22px;display:flex;
    flex-direction:column;gap:10px}
  .dep-stage:fullscreen .dep-wrap{flex:1;min-height:0;display:flex;align-items:center;
    justify-content:center;overflow:auto;margin:0}
  .dep-stage:-webkit-full-screen{background:var(--bg);padding:18px 22px;display:flex;
    flex-direction:column;gap:10px}
  .dep-detail{margin:0 0 8px}
  .dep-detail table{margin:0}
  g.node{cursor:pointer}
  g.node.dimmed{opacity:.15}
  .dep-edge-dim{opacity:.07}
  .dep-edge-in{stroke:#5ec8ff !important;stroke-width:2.6px !important}
  .dep-edge-out{stroke:#1fd27a !important;stroke-width:2.6px !important}
  g.node.sel rect{stroke:#1fd27a !important;stroke-width:2.5px !important}
"""

JS = """var DEP_ROLE = __ROLE__;
var DEP_LABEL = __LABEL__;

(function wireGraph() {
  var svg = document.getElementById("depsvg");
  if (!svg) return;

  var nodes = {};
  Array.prototype.forEach.call(svg.querySelectorAll("g.node"), function (n) {
    var m = /flowchart-([A-Za-z0-9]+)-\\d+$/.exec(n.id || "");
    if (m) { nodes[m[1]] = n; }
  });

  var edges = [];
  Array.prototype.forEach.call(svg.querySelectorAll("g.edgePaths > *"), function (e) {
    var raw = e.getAttribute("data-id") || (e.id || "").replace(/^depsvg-/, "");
    var m = /^L_([A-Za-z0-9]+)_([A-Za-z0-9]+)_\\d+$/.exec(raw);
    if (m) { edges.push({ el: e, from: m[1], to: m[2] }); }
  });

  function clearAll() {
    Object.keys(nodes).forEach(function (k) {
      nodes[k].classList.remove("dimmed", "sel");
    });
    edges.forEach(function (e) {
      e.el.classList.remove("dep-edge-dim", "dep-edge-in", "dep-edge-out");
    });
    document.getElementById("dep-detail").innerHTML = "";
    document.getElementById("dep-status").textContent = "Nothing selected";
  }

  function nameOf(k) { return DEP_LABEL[k] || k; }

  function fmt(list) {
    if (!list.length) { return '<span class="dim-note">none</span>'; }
    return list.map(function (k) { return "<code>" + nameOf(k) + "</code>"; }).join(" ");
  }

  function select(key) {
    var imports = [], importedBy = [], keep = {};
    keep[key] = 1;
    edges.forEach(function (e) {
      if (e.from === key) { imports.push(e.to); keep[e.to] = 1; }
      else if (e.to === key) { importedBy.push(e.from); keep[e.from] = 1; }
    });
    Object.keys(nodes).forEach(function (k) {
      if (keep[k]) { nodes[k].classList.remove("dimmed"); }
      else { nodes[k].classList.add("dimmed"); }
      if (k === key) { nodes[k].classList.add("sel"); }
      else { nodes[k].classList.remove("sel"); }
    });
    edges.forEach(function (e) {
      e.el.classList.remove("dep-edge-dim", "dep-edge-in", "dep-edge-out");
      if (e.from === key) { e.el.classList.add("dep-edge-out"); }
      else if (e.to === key) { e.el.classList.add("dep-edge-in"); }
      else { e.el.classList.add("dep-edge-dim"); }
    });
    document.getElementById("dep-status").textContent =
      nameOf(key) + ": imports " + imports.length + ", imported by " + importedBy.length;
    document.getElementById("dep-detail").innerHTML =
      "<table><tr><th>" + nameOf(key) + "</th><th>Modules</th></tr>" +
      "<tr><td>Imports, outgoing arrows in green</td><td>" + fmt(imports) + "</td></tr>" +
      "<tr><td>Imported by, incoming arrows in blue</td><td>" + fmt(importedBy) + "</td></tr>" +
      (DEP_ROLE[key] ? "<tr><td>Role</td><td>" + DEP_ROLE[key] + "</td></tr>" : "") +
      "</table>";
  }

  Object.keys(nodes).forEach(function (k) {
    nodes[k].addEventListener("click", function () { select(k); });
  });
  document.getElementById("dep-reset").addEventListener("click", clearAll);

  var wrap = document.getElementById("dep-stage");
  var fsBtn = document.getElementById("dep-full");
  function inFullscreen() {
    return document.fullscreenElement === wrap || document.webkitFullscreenElement === wrap;
  }
  fsBtn.addEventListener("click", function () {
    if (inFullscreen()) {
      (document.exitFullscreen || document.webkitExitFullscreen).call(document);
    } else {
      var req = wrap.requestFullscreen || wrap.webkitRequestFullscreen;
      if (req) { req.call(wrap); }
    }
  });
  ["fullscreenchange", "webkitfullscreenchange"].forEach(function (ev) {
    document.addEventListener(ev, function () {
      fsBtn.textContent = inFullscreen() ? "Exit full screen" : "Full screen";
    });
  });
})();

var buttons = document.querySelectorAll("nav button");"""

s = io.open(PAGE, encoding="utf-8").read()

# Idempotent: strip any previously injected graph so re-running cannot stack two
# copies of a 90 KB SVG into the page.
if '<section id="deps">' in s:
    a = s.index('<section id="deps">')
    b = s.index('<section id="flow">', a)
    s = s[:a] + s[b:]
    nav_old = chr(60) + 'button data-t="deps">Dependency graph</button>' + chr(10) + "  "
    s = s.replace(nav_old, "", 1)
    print("removed a previous graph section")

assert '<button data-t="flow">Verification flow</button>' in s
s = s.replace('<button data-t="flow">Verification flow</button>',
              '<button data-t="deps">Dependency graph</button>\n  '
              '<button data-t="flow">Verification flow</button>', 1)

assert '<section id="flow">' in s
s = s.replace('<section id="flow">', SECTION.replace("__SVG__", svg) + '<section id="flow">', 1)

assert "  @media(max-width:820px){" in s
s = s.replace("  @media(max-width:820px){", CSS + "  @media(max-width:820px){", 1)

assert 'var buttons = document.querySelectorAll("nav button");' in s
s = s.replace('var buttons = document.querySelectorAll("nav button");',
              JS.replace("__ROLE__", json.dumps(ROLE)).replace("__LABEL__", json.dumps(LABELS)), 1)

io.open(PAGE, "w", encoding="utf-8", newline="\n").write(s)
print("dependency graph injected, page is now", round(len(s) / 1024), "KB")

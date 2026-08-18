"""Replace the module-map card data with full per-file detail.

Exports are taken from the source, and the imports / imported-by lists come from
the same parsed graph the dependency diagram is built from, so nothing here is
written from memory.
"""
import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.environ.get("NETRAID_BUILD_DIR", os.path.join(ROOT, "build"))
PAGE = os.path.join(ROOT, "docs", "index.html")
SRC = os.path.join(ROOT, "app", "src")

graph = json.load(io.open(os.path.join(SP, "graph.json"), encoding="utf-8"))
edges = graph["edges"]
ext = graph["ext"]

imports_of = {k: sorted(v) for k, v in edges.items()}
imported_by = {}
for src, tgts in edges.items():
    for t in tgts:
        imported_by.setdefault(t, []).append(src)
for k in imported_by:
    imported_by[k] = sorted(imported_by[k])


def short(path):
    return path.split("/")[-1]


def exports_of(rel):
    p = os.path.join(SRC, rel)
    if not os.path.exists(p):
        return []
    text = io.open(p, encoding="utf-8").read()
    found = re.findall(
        r"^export (?:async )?(?:function|const|class|interface|type|enum) ([A-Za-z0-9_]+)",
        text, re.M)
    seen, out = set(), []
    for f in found:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


# What each file is FOR, and the thing about it worth knowing. Written per file;
# the exports and edges below are extracted, not recalled.
NOTES = {
    "netraid/index.ts": (
        "Public API and the verdict.",
        "The only surface a host application touches. Aggregates the three verify captures "
        "with a margin rule so a near-miss against a second person cannot pass, applies the "
        "enrollment anti-spoof gate, writes the attendance row, and starts the sync engine. "
        "Throws DuplicateFaceError when a face is already enrolled under a different id, "
        "enforcing one face to one identity, and SpoofedEnrollmentError when the enrollment "
        "gate is armed and fails."),
    "netraid/frameProcessor.ts": (
        "The real-time worklet. The largest and most performance-sensitive file.",
        "Runs on a dedicated thread for every second camera frame; the rest are dropped to "
        "bound worklet memory, which otherwise fragments into an OutOfMemoryError after about "
        "a minute of continuous processing. Owns the 1.2 s sensor warm-up, detection, "
        "landmarks, the challenge state machine, continuity binding, the passive anti-spoof "
        "read, the post-flash colour-recovery gate, and the frontality, sharpness and "
        "confidence gates. Emits only aligned 112x112 crops that passed all of them. The "
        "anti-spoof crop geometry lives here: 2.7x the face box, padded rather than clamped, "
        "fed as BGR in [0,1]."),
    "netraid/liveness.ts": (
        "Challenge geometry and the combined liveness verdict.",
        "Blink, smile and head-turn are computed from landmark ratios: eye aspect ratio, mouth "
        "width against a learned neutral, and a yaw ratio. Each step requires a confirmed "
        "neutral face after its prompt, then the gesture no sooner than MIN_REACTION_MS and no "
        "later than the window closes, so a recording cannot arrive mid-gesture and cannot be "
        "run until the demanded gesture comes around. decideLiveness combines the active "
        "result with each passive gate according to its configured mode."),
    "netraid/types.ts": (
        "Every type, and DEFAULT_CONFIG.",
        "The single place an operating point changes. Each threshold carries the measurement "
        "it came from in a comment, and each gate carries a mode of off, report or enforce, so "
        "a gate can be measured on real hardware before it is allowed to reject anyone."),
    "netraid/recognition.ts": (
        "Face embedding.",
        "Loads MobileFaceNet through the useTensorflowModel hook, which is the path that works "
        "reliably on device, and produces a 512-d vector. embedTTA also embeds the mirrored "
        "crop and averages, which measurably steadies the score on off-angle faces. float32 "
        "rather than int8 or fp16: int8 emitted NaN and fp16 stalled the interpreter on the "
        "target handset."),
    "netraid/align.ts": (
        "Face alignment.",
        "Umeyama similarity transform from 5 landmarks onto a fixed reference, then a warp to "
        "112x112. Alignment is what lets one template match a face at a different angle and "
        "distance; without it the embedding shifts with pose."),
    "netraid/math.ts": (
        "Matching arithmetic. No dependencies beyond types.",
        "Cosine similarity, L2 normalisation, and the aggregations the verdict relies on: "
        "median across a capture burst, and consensus, the value at least N captures reached. "
        "robustAverage builds an enrollment template while discarding embedding outliers, so "
        "one bad crop in six cannot poison an identity."),
    "netraid/store.ts": (
        "Encrypted local storage.",
        "op-sqlite with SQLCipher, AES-256 page encryption. Holds face templates as embeddings "
        "only and the attendance queue. No image is ever written to disk. The queue tracks "
        "sync attempts so a record that keeps failing can be seen rather than silently lost."),
    "netraid/keys.ts": (
        "Hardware-backed key management.",
        "The SQLCipher key is generated once on the device and stored in the Android Keystore "
        "or the iOS Keychain. It never leaves the handset and is never derived from anything "
        "the user types, so a stolen phone yields an encrypted database and nothing else."),
    "netraid/sync.ts": (
        "Offline-first sync and purge.",
        "Writes are always local first. Flushes the queue when NetInfo reports reachability, "
        "batching records that each carry a client-generated UUID. Only ids the server "
        "acknowledged are purged locally; anything unacknowledged stays queued, so a dropped "
        "connection loses nothing and a retry cannot duplicate a record."),
    "netraid/chroma.ts": (
        "Screen-illumination challenge. Off by default.",
        "Flashes a colour sequence drawn from a CSPRNG at verification time, so it cannot have "
        "been recorded, and measures how the face responds. Sound physics, kept with its tests "
        "for indoor deployments that calibrate it, but off by default: it needs the phone "
        "screen to be a meaningful share of the light, which rules out daylight, and it did "
        "not separate a live face from a replay on the target handset."),
    "netraid/quality.ts": (
        "Capture quality gate.",
        "Laplacian-variance sharpness and a brightness band on the aligned crop. A blurred or "
        "badly exposed crop degrades an enrollment template permanently and makes a "
        "verification probe unreliable, so such crops are dropped before they reach either."),
    "netraid/blazeface.ts": (
        "Detector post-processing.",
        "Anchor generation, box and keypoint decoding, and non-maximum suppression over the "
        "raw BlazeFace output, then picks the primary face. Kept separate from the worklet so "
        "it can be unit-tested without a camera."),
    "netraid/calibration.ts": (
        "Calibration telemetry.",
        "One machine-readable line per attempt, emitted in release builds, which is how a "
        "deployment reads its own operating point off its own handsets with nothing but a USB "
        "cable. Carries model scores and timings only: never an embedding, never an image, "
        "never a person id."),
    "netraid/imaging.ts": (
        "Pure raster helpers.",
        "Resize, crop and normalisation used by the camera-free path. Being pure functions "
        "with no native dependency is what makes the core algorithms testable on a desktop."),
    "netraid/screenLamp.ts": (
        "Screen brightness bridge, Android only.",
        "Raises this window's brightness for the duration of the chroma flash and hands it "
        "straight back. A per-window override, so no system setting changes, no permission is "
        "needed, and Android releases it if the app loses focus, meaning a crash cannot strand "
        "a field handset at full brightness. Called behind a platform guard, so iOS is "
        "unaffected."),
    "netraid/demoPipeline.ts": (
        "Camera-free pipeline demo.",
        "Runs detection, alignment, embedding and matching over bundled frames, so the full "
        "pipeline can be demonstrated and timed without a live camera or a live subject."),
    "netraid/demoAssets.ts": (
        "Generated demo frames.",
        "Produced by ml/scripts/make_demo_assets.py. Not hand-edited."),
    "screens/HomeScreen.tsx": (
        "Dashboard.",
        "Enrolled count, pending sync count, connectivity and store status, and the entry "
        "points to enrollment and verification."),
    "screens/EnrollScreen.tsx": (
        "Enrollment, six-shot capture.",
        "Captures without a gesture challenge, because enrollment needs a clean frontal face "
        "rather than a reaction test, then hands the burst to enroll(), which keeps the "
        "sharpest crops and averages them into one template."),
    "screens/VerifyScreen.tsx": (
        "Verification.",
        "Drives the challenge prompts, the optional chroma flash, capture collection and the "
        "verdict card. Nothing on screen claims a pass until every barrier has ruled: the card "
        "between the gestures and the verdict reads as checking, not as passed, because "
        "showing green and then rejecting reads as a broken app even when the rejection is "
        "correct."),
    "screens/PipelineDemoScreen.tsx": (
        "Pipeline demo.",
        "Runs demoPipeline over bundled frames and shows each stage with its timing, for "
        "demonstrating the pipeline without a camera."),
}

cards = []
for rel, (role, detail) in NOTES.items():
    path = os.path.join(SRC, rel)
    lines = len(io.open(path, encoding="utf-8").read().split("\n")) if os.path.exists(path) else 0
    cards.append({
        "f": short(rel),
        "dir": rel.split("/")[0],
        "n": lines,
        "role": role,
        "detail": detail,
        "ex": exports_of(rel),
        "imp": [short(x) for x in imports_of.get(rel, [])],
        "by": [short(x) for x in imported_by.get(rel, [])],
        "lib": [x for x in ext.get(rel, []) if x != "react"],
    })

page = io.open(PAGE, encoding="utf-8").read()

start = page.index("var MODULES = [")
end = page.index("});", page.index("grid.appendChild(d);") - 400)
end = page.index("\n", page.index("grid.appendChild(d);")) + 1
end = page.index("});", end) + 4

new_js = "var MODULES = " + json.dumps(cards, ensure_ascii=False) + """;

var grid = document.getElementById("modgrid");
MODULES.forEach(function (m) {
  var d = document.createElement("div");
  d.className = "card";
  d.innerHTML = '<span class="n">' + m.n + ' lines</span><h4>' + m.f + "</h4><p>" + m.role + "</p>";
  d.addEventListener("click", function () {
    var list = function (arr) {
      return arr.length
        ? arr.map(function (x) { return "<code>" + x + "</code>"; }).join(" ")
        : '<span class="dim-note">none</span>';
    };
    document.getElementById("dlg-t").textContent = "app/src/" + m.dir + "/" + m.f;
    document.getElementById("dlg-b").innerHTML =
      "<p>" + m.detail + "</p>" +
      "<table>" +
      "<tr><td>Exports</td><td>" + list(m.ex) + "</td></tr>" +
      "<tr><td>Imports</td><td>" + list(m.imp) + "</td></tr>" +
      "<tr><td>Imported by</td><td>" + list(m.by) + "</td></tr>" +
      "<tr><td>Libraries</td><td>" + list(m.lib) + "</td></tr>" +
      "<tr><td>Size</td><td>" + m.n + " lines</td></tr>" +
      "</table>";
    document.getElementById("dlg").showModal();
  });
  grid.appendChild(d);
});"""

page = page[:start] + new_js + page[end:]
io.open(PAGE, "w", encoding="utf-8", newline="\n").write(page)
print("module detail rebuilt for", len(cards), "files")

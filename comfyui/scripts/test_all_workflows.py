#!/usr/bin/env python3
"""
test_all_workflows.py — End-to-end test of all comfy_graph builders.

Runs each workflow, records success/failure/OOM, logs results to test_results.json.
For video, tests increasing lengths to find the OOM threshold.

Usage: python test_all_workflows.py [--output-dir ./test_outputs] [--skip-second-pass]
"""

import json
import os
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import comfy_graph as cg

BASE = os.environ.get("COMFY_URL", "http://localhost:8188").rstrip("/")
OUTPUT_DIR = Path("test_outputs")
RESULTS_FILE = Path("test_results.json")
SKIP_SECOND_PASS = "--skip-second-pass" in sys.argv

results = []


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def submit(workflow):
    payload = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(f"{BASE}/prompt", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def poll(prompt_id, timeout=900):
    """Poll until done. Returns (status_str, outputs, elapsed_s, error_msg)."""
    start = time.time()
    deadline = start + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(f"{BASE}/history/{prompt_id}", timeout=15) as r:
            hist = json.load(r)
        entry = hist.get(prompt_id)
        if entry:
            outputs = entry.get("outputs", {})
            status = entry.get("status", {})
            st = status.get("status_str") or status.get("status", "")
            msgs = status.get("messages", [])
            oom_msg = ""
            for m in msgs:
                if isinstance(m, list) and len(m) >= 2:
                    txt = str(m[1])
                    if "out of memory" in txt.lower() or "cuda out" in txt.lower():
                        oom_msg = txt[:200]
            if oom_msg:
                return "oom", {}, round(time.time() - start, 1), oom_msg
            if outputs:
                assets = []
                for nout in outputs.values():
                    for v in nout.values():
                        if isinstance(v, list):
                            for item in v:
                                if isinstance(item, dict) and "filename" in item:
                                    assets.append(item["filename"])
                return "success", assets, round(time.time() - start, 1), ""
            if status.get("completed") is True and st == "success":
                # All nodes cached — workflow is valid, output already exists from prior run
                cached_nodes = next(
                    (m[1].get("nodes", []) for m in msgs
                     if isinstance(m, list) and len(m) >= 2 and m[0] == "execution_cached"),
                    [])
                if cached_nodes:
                    return "cached", ["[all nodes cached from prior run]"], round(time.time() - start, 1), ""
                return "error", {}, round(time.time() - start, 1), "completed with no outputs"
            if st == "error" or (status.get("completed") is False and st == "error"):
                err = str(msgs[-1]) if msgs else st
                return "error", {}, round(time.time() - start, 1), err
        time.sleep(3)
    return "timeout", {}, timeout, f"Did not complete in {timeout}s"


def save_assets(prompt_id, output_dir):
    try:
        with urllib.request.urlopen(f"{BASE}/history/{prompt_id}", timeout=15) as r:
            hist = json.load(r)
        entry = hist.get(prompt_id, {})
        saved = []
        for nout in entry.get("outputs", {}).values():
            for v in nout.values():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict) and "filename" in item:
                            fname = item["filename"]
                            sf = item.get("subfolder", "")
                            tp = item.get("type", "output")
                            url = f"{BASE}/view?filename={urllib.parse.quote(fname)}&subfolder={sf}&type={tp}&_={int(time.time()*1000)}"
                            dest = output_dir / fname
                            with urllib.request.urlopen(url, timeout=120) as r:
                                dest.write_bytes(r.read())
                            saved.append(str(dest))
        return saved
    except Exception as e:
        return [f"save_error: {e}"]


def run_test(name, workflow, timeout=900):
    log(f"  Submitting: {name}")
    try:
        result = submit(workflow)
        node_errors = result.get("node_errors", {})
        if node_errors:
            log(f"  Node errors: {node_errors}")
            rec = {"name": name, "status": "node_error", "errors": str(node_errors), "elapsed_s": 0, "assets": []}
            results.append(rec)
            save_results()
            return rec
        pid = result["prompt_id"]
        log(f"  prompt_id: {pid} — waiting (timeout={timeout}s)...")
        status, assets, elapsed, err = poll(pid, timeout=timeout)
        log(f"  → {status} in {elapsed}s assets={assets}")
        saved = []
        if status in ("success", "cached"):
            saved = save_assets(pid, OUTPUT_DIR)
            if status == "cached":
                status = "success"  # treat as pass for downstream tests
        rec = {"name": name, "status": status, "elapsed_s": elapsed,
               "assets": assets, "saved": saved, "error": err}
    except Exception as e:
        log(f"  EXCEPTION: {e}")
        rec = {"name": name, "status": "exception", "error": str(e), "elapsed_s": 0, "assets": []}
    results.append(rec)
    save_results()
    return rec


def upload_to_input(local_path):
    """Upload a local file to ComfyUI's input directory. Returns the uploaded filename."""
    fname = Path(local_path).name
    data = Path(local_path).read_bytes()
    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{fname}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"subfolder\"\r\n\r\n\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{BASE}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.load(r)
    return resp["name"]


def save_results():
    RESULTS_FILE.write_text(json.dumps(results, indent=2))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = int(time.time()) % (2 ** 31)

    log("=" * 60)
    log("ComfyUI workflow test suite")
    log("=" * 60)

    # ── 1. Flux2 text-to-image ────────────────────────────────────
    log("\n[1] Flux2 t2i")
    run_test("flux2_t2i",
        cg.flux2_text_to_image("a red apple on a wooden table, photorealistic",
                               width=512, height=512, seed=seed))

    # ── 2. Flux2 t2i + LoRA ──────────────────────────────────────
    log("\n[2] Flux2 t2i + pixel art LoRA")
    run_test("flux2_t2i_lora",
        cg.flux2_text_to_image("a red apple", width=512, height=512, seed=seed,
                               lora="pixel_art_style_z_image_turbo.safetensors",
                               lora_strength=1.0,
                               filename_prefix="test_t2i_lora"))

    # ── 3. Flux2 single-image edit ────────────────────────────────
    log("\n[3] Flux2 i2i (using t2i output as ref)")
    ref_img = None
    t2i_rec = next((r for r in results if r.get("name") == "flux2_t2i" and r.get("status") == "success"), None)
    if t2i_rec:
        saved_pngs = [s for s in t2i_rec.get("saved", []) if s.endswith(".png")]
        if saved_pngs:
            try:
                ref_img = upload_to_input(saved_pngs[0])
                log(f"  Uploaded reference image: {ref_img}")
            except Exception as e:
                log(f"  Upload failed: {e}")
    if ref_img:
        run_test("flux2_i2i",
            cg.flux2_single_image_edit(ref_img, "same scene at night",
                                       width=512, height=512, seed=seed,
                                       filename_prefix="test_i2i"))
    else:
        results.append({"name": "flux2_i2i", "status": "skipped", "error": "no ref image"})
        save_results()

    # ── 4. Flux2 multiple angles ──────────────────────────────────
    log("\n[4] Flux2 multiple angles")
    if ref_img:
        run_test("flux2_angles",
            cg.flux2_multiple_angles(ref_img,
                                      angle_prompts=["front view", "side view", "3/4 angle view"],
                                      filename_prefix="test_angles"),
            timeout=180)
    else:
        results.append({"name": "flux2_angles", "status": "skipped"})
        save_results()

    # ── 5. TTS ────────────────────────────────────────────────────
    log("\n[5] Qwen TTS")
    run_test("qwen_tts",
        cg.qwen_tts("This is a test of the text to speech system.",
                    voice_instruct="Clear, neutral male voice",
                    filename_prefix="test_tts"),
        timeout=120)

    # ── 6. LTX2 t2v — length sweep ───────────────────────────────
    log("\n[6] LTX2 t2v — length sweep (2s, 4s, 6s, 8s, 10s)")
    prompt_t2v = "a glowing blue jellyfish drifting through dark ocean water, cinematic"
    t2v_oom_at = None
    for secs in [2, 4, 6, 8, 10]:
        if t2v_oom_at:
            log(f"  Skipping {secs}s (OOM at {t2v_oom_at}s)")
            results.append({"name": f"ltx2_t2v_{secs}s", "status": "skipped",
                           "error": f"OOM at {t2v_oom_at}s"})
            save_results()
            continue
        rec = run_test(f"ltx2_t2v_{secs}s",
                       cg.ltx2_text_to_video(prompt_t2v, seconds=secs,
                                              filename_prefix=f"test_t2v_{secs}s", seed=seed),
                       timeout=900)
        if rec["status"] in ("oom", "error"):
            t2v_oom_at = secs
            log(f"  !! OOM/error at {secs}s — stopping length sweep")

    # ── 7. LTX2 i2v — 3s with camera LoRA ───────────────────────
    log("\n[7] LTX2 i2v 3s + dolly-in camera LoRA")
    i2v_ref = ref_img  # already uploaded, or None
    if i2v_ref:
        run_test("ltx2_i2v_3s_dolly",
            cg.ltx2_image_to_video(i2v_ref, "cinematic dolly in, dramatic lighting",
                                    seconds=3, camera_lora="dolly-in",
                                    filename_prefix="test_i2v_dolly", seed=seed),
            timeout=900)
    else:
        results.append({"name": "ltx2_i2v_3s_dolly", "status": "skipped"})
        save_results()

    # ── 8. LTX2 i2v — length sweep ───────────────────────────────
    log("\n[8] LTX2 i2v — length sweep (3s, 5s, 8s, 12s)")
    if i2v_ref:
        i2v_oom_at = None
        for secs in [3, 5, 8, 12]:
            if i2v_oom_at:
                results.append({"name": f"ltx2_i2v_{secs}s", "status": "skipped",
                               "error": f"OOM at {i2v_oom_at}s"})
                save_results()
                continue
            rec = run_test(f"ltx2_i2v_{secs}s",
                           cg.ltx2_image_to_video(i2v_ref, prompt_t2v, seconds=secs,
                                                  filename_prefix=f"test_i2v_{secs}s", seed=seed),
                           timeout=900)
            if rec["status"] in ("oom", "error"):
                i2v_oom_at = secs
    else:
        log("  Skipping — no reference frame")

    # ── 9. LTX2 ingredients (reference-sheet IC-LoRA) ────────────
    # Smoke test: feeds an image as a degenerate single-element "sheet"
    # (looped in-graph to a static reference video) and checks the
    # ingredients IC-LoRA graph runs without node errors / OOM. Real sheets
    # are multi-panel composites — see make_reference_sheet.py /
    # storyboard generate_reference_sheet.py.
    log("\n[9] LTX2 ingredients (reference-sheet IC-LoRA, 3s @ 768x448)")
    if i2v_ref:
        run_test("ltx2_ingredients_3s",
            cg.ltx2_ingredients_to_video(i2v_ref,
                                         "the subject in a sunlit scene, cinematic",
                                         seconds=3, width=768, height=448,
                                         filename_prefix="test_ingredients", seed=seed),
            timeout=900)
    else:
        results.append({"name": "ltx2_ingredients_3s", "status": "skipped",
                        "error": "no ref image"})
        save_results()

    # ── Summary ───────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    counts = {}
    for r in results:
        st = r.get("status", "?")
        counts[st] = counts.get(st, 0) + 1
        elapsed = r.get("elapsed_s", "")
        elapsed_str = f" ({elapsed}s)" if elapsed else ""
        err = r.get("error", "")
        err_str = f" — {err[:80]}" if err else ""
        log(f"  {r['name']:40} {st}{elapsed_str}{err_str}")

    log(f"\nTotal: {len(results)} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    log(f"Results saved to: {RESULTS_FILE.resolve()}")
    save_results()


if __name__ == "__main__":
    main()

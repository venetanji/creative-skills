#!/usr/bin/env python3
"""Offline regressions; --live --video PATH also checks real GLB/MP4 outputs."""
import argparse
import io
import json
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import comfy_graph
import core
import sam3d_body
import sam3d_models


def nodes_of_type(workflow, class_type):
    return [node for node in workflow.values() if node["class_type"] == class_type]


class SAM3DBodyTests(unittest.TestCase):
    def test_full_workflow(self):
        workflow = sam3d_body.sam3d_video_to_body("person.mp4")
        for class_type in ("SAM3_VideoTrack", "RTDETR_detect", "MoGeInference",
                           "SAM3DBody_Predict", "SAM3DBody_FaceExpression",
                           "SAM3DBody_Smooth", "BuildPoseFile", "Preview3D",
                           "SAM3DBody_Render", "SaveVideo"):
            self.assertEqual(len(nodes_of_type(workflow, class_type)), 1)
        components = next(key for key, node in workflow.items()
                          if node["class_type"] == "GetVideoComponents")
        self.assertEqual(nodes_of_type(workflow, "CreateVideo")[0]["inputs"]["fps"],
                         [components, 2])
        self.assertEqual(nodes_of_type(workflow, "BuildPoseFile")[0]["inputs"]["fps"],
                         [components, 2])
        for node_id, node in workflow.items():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertIn(value[0], workflow)
                    self.assertLess(int(value[0]), int(node_id))

    def test_low_memory_workflow(self):
        workflow = sam3d_body.sam3d_video_to_body(
            "person.mp4", seconds=0, batch_size=1, tracking=False, moge=False,
            face_expression=False, hand_refinement=False, overlay=False, keep_audio=False)
        for class_type in ("Video Slice", "SAM3_VideoTrack", "MoGeInference",
                           "SAM3DBody_FaceExpression"):
            self.assertFalse(nodes_of_type(workflow, class_type))
        prediction = nodes_of_type(workflow, "SAM3DBody_Predict")[0]["inputs"]
        self.assertEqual(prediction["batch_size"], 1)
        self.assertFalse(prediction["run_hand_refinement"])
        self.assertNotIn("track_data", prediction)
        self.assertEqual(prediction["fov"], 0.0)
        self.assertNotIn("background", nodes_of_type(workflow, "SAM3DBody_Render")[0]["inputs"])
        self.assertNotIn("audio", nodes_of_type(workflow, "CreateVideo")[0]["inputs"])

    def test_dynamic_inputs(self):
        for style in ("body_mesh", "scail"):
            workflow = sam3d_body.sam3d_video_to_body("person.mp4", export_style=style)
            inputs = nodes_of_type(workflow, "BuildPoseFile")[0]["inputs"]
            self.assertEqual(inputs["format"], "glb")
            self.assertEqual(inputs["format.mesh_style"], style)
            self.assertIn("format.bone_smooth_window", inputs)
            self.assertEqual(nodes_of_type(workflow, "SAM3DBody_Render")[0]["inputs"]
                             ["render_style.shader"], "default")

    def test_invalid_options(self):
        for options in ({"video_filename": ""}, {"seconds": -1}, {"start_time": -1},
                        {"batch_size": 0}, {"moge_batch_size": 65},
                        {"max_people": 0}, {"detection_threshold": 1.1},
                        {"export_style": "unknown"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                sam3d_body.sam3d_video_to_body(**{"video_filename": "person.mp4", **options})

    def test_routing(self):
        with patch.dict(os.environ, {"COMFY_URL_FLUX": "http://image:8188",
                                     "COMFY_URL_VIDEO": "http://video:8188/"}):
            self.assertEqual(comfy_graph._resolve_base_url("sam3d"), "http://video:8188")
            self.assertEqual(comfy_graph._resolve_base_url("t2i"), "http://image:8188")

    def test_dump_does_not_upload(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            with patch.object(comfy_graph, "DUMP_ONLY", True), patch.object(core, "upload_image") as upload:
                workflow = comfy_graph._h_sam3d({"video": video.name}, None, "")
                upload.assert_not_called()
                self.assertEqual(nodes_of_type(workflow, "LoadVideo")[0]["inputs"]["file"], video.name)

    def test_local_video_upload(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            with patch.object(comfy_graph, "DUMP_ONLY", False), patch.object(
                    core, "upload_image", return_value="uploaded.mp4") as upload:
                workflow = comfy_graph._h_sam3d({"video": video.name}, None, "")
                upload.assert_called_once_with(video.name)
                self.assertEqual(nodes_of_type(workflow, "LoadVideo")[0]["inputs"]["file"], "uploaded.mp4")

    def test_upload_failure_is_not_silenced(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            with patch.object(comfy_graph, "DUMP_ONLY", False), patch.object(
                    core, "upload_image", side_effect=OSError("upload failed")):
                with self.assertRaises(OSError):
                    comfy_graph._h_sam3d({"video": video.name}, None, "")

    def test_download_preview_glb_and_video(self):
        entry = {"outputs": {"input": {"images": [{"filename": "input.mp4", "type": "input"}]},
                             "mesh": {"result": ["preview3d_test.glb", None, None]},
                             "video": {"images": [{"filename": "overlay.mp4",
                                                    "subfolder": "video/SAM3D_body", "type": "output"}]}}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.dict(os.environ, {"OPENCLAW_MEDIA_DIR": str(output / "outbound")}), patch.object(
                    core.urllib.request, "urlopen", side_effect=lambda *args, **kwargs: io.BytesIO(b"asset")) as fetch:
                core._save_assets(entry, output)
            self.assertEqual((output / "preview3d_test.glb").read_bytes(), b"asset")
            self.assertEqual((output / "overlay.mp4").read_bytes(), b"asset")
            self.assertEqual(fetch.call_count, 2)

    def test_execution_error_with_partial_outputs_raises(self):
        responses = [{"prompt_id": "test-id"}, {"test-id": {
            "outputs": {"mesh": {"result": ["partial.glb"]}},
            "status": {"status_str": "error", "messages": [["execution_error", {
                "exception_message": "CUDA out of memory"}]]}}}]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(core.urllib.request, "urlopen", side_effect=[
                    io.BytesIO(json.dumps(response).encode()) for response in responses]):
                with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                    core._submit_and_wait({}, Path(directory), timeout=1)

    def test_model_validation(self):
        header = json.dumps({"tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        complete = struct.pack("<Q", len(header)) + header + b"\0" * 4
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "test.safetensors"
            model.write_bytes(complete)
            self.assertEqual(sam3d_models.validate_safetensors(model), len(complete))
            for data in (complete[:-1], b"", b"<!DOCTYPE html>"):
                model.write_bytes(data)
                with self.assertRaises(ValueError):
                    sam3d_models.validate_safetensors(model)


def run_live(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    core.BASE = args.url or comfy_graph._resolve_base_url("sam3d")
    comfy_graph.DUMP_ONLY = False
    options = {"video": args.video, "seconds": args.seconds,
               "batch_size": args.batch_size, "moge_batch_size": 1,
               "prefix": "video/SAM3D_body_test"}
    if args.low_memory:
        options.update(no_tracking=True, no_moge=True, no_face=True, no_hand_refinement=True)
    workflow = comfy_graph._h_sam3d(options, None, "person")
    (output / "workflow.json").write_text(json.dumps(workflow, indent=2) + "\n")
    with patch.dict(os.environ, {"OPENCLAW_MEDIA_DIR": str(output)}):
        prompt_id = core._submit_and_wait(workflow, output, timeout=args.timeout)
    with urllib.request.urlopen(f"{core.BASE}/history/{prompt_id}", timeout=30) as response:
        entry = json.load(response)[prompt_id]
    (output / "history.json").write_text(json.dumps(entry, indent=2) + "\n")
    if entry["status"]["status_str"] != "success":
        raise AssertionError(f"Execution did not succeed: {entry['status']}")
    mesh_output = next(value for value in entry["outputs"].values() if "result" in value)
    mesh = output / Path(mesh_output["result"][0]).name
    with mesh.open("rb") as stream:
        magic, version, size = struct.unpack("<4sII", stream.read(12))
        if magic != b"glTF" or version != 2 or size != mesh.stat().st_size:
            raise AssertionError("Invalid GLB header or incomplete download")
        chunk_size, chunk_type = struct.unpack("<I4s", stream.read(8))
        if chunk_type != b"JSON":
            raise AssertionError("GLB is missing its JSON chunk")
        document = json.loads(stream.read(chunk_size))
    if not document.get("meshes") or not document.get("animations"):
        raise AssertionError("GLB is missing meshes or animations")
    if not document.get("skins") or not document["skins"][0].get("joints"):
        raise AssertionError("Body GLB is missing its skeleton")
    keyframes = max(document["accessors"][sampler["input"]]["count"]
                    for animation in document["animations"] for sampler in animation["samplers"])
    video_files = [output / Path(item["filename"]).name for value in entry["outputs"].values()
                   for items in value.values() if isinstance(items, list)
                   for item in items if isinstance(item, dict) and item.get("type") == "output"
                   and item.get("filename", "").endswith(".mp4")]
    if not video_files or not all(video.is_file() and video.stat().st_size > 1000 for video in video_files):
        raise AssertionError("Missing or empty overlay MP4")
    video_stats = []
    if shutil.which("ffprobe"):
        for video in video_files:
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-count_frames", "-show_entries",
                "stream=codec_name,codec_type,width,height,r_frame_rate,duration,nb_read_frames",
                "-of", "json", str(video)], check=True, capture_output=True, text=True)
            streams = json.loads(probe.stdout)["streams"]
            frames = next(stream for stream in streams if stream["codec_type"] == "video")
            if int(frames["nb_read_frames"]) < 2:
                raise AssertionError("Overlay video has fewer than two decodable frames")
            if int(frames["nb_read_frames"]) != keyframes:
                raise AssertionError("GLB animation and overlay video have different frame counts")
            video_stats.append({"file": str(video), "streams": streams})
    result = {"prompt_id": prompt_id, "server": core.BASE, "status": "success",
              "glb": str(mesh), "glb_bytes": mesh.stat().st_size,
              "meshes": len(document["meshes"]), "animations": len(document["animations"]),
              "joints": len(document["skins"][0]["joints"]), "keyframes": keyframes,
              "videos": [str(video) for video in video_files],
              "video_stats": video_stats, "options": options}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--video")
    parser.add_argument("--url")
    parser.add_argument("--seconds", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("test_outputs/sam3d"))
    args = parser.parse_args()
    if args.live and not args.video:
        parser.error("--live requires --video (local path or ComfyUI input name)")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SAM3DBodyTests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
    if args.live:
        run_live(args)


if __name__ == "__main__":
    main()

import json
import os
import subprocess
import sys

import pytest

from creative_comfy_graph.moss_image import MODEL_PATH, moss_image_text_to_image


def inputs(graph, kind):
    return next(node["inputs"] for node in graph.values() if node["class_type"] == kind)


def test_upstream_topology_and_low_vram_defaults():
    graph = moss_image_text_to_image("A lighthouse", seed=0)
    assert len(graph) == 9
    assert inputs(graph, "MOSSImageHFLoader") == {
        "model_path": MODEL_PATH, "weight_dtype": "fp8_e4m3fn", "text_device": "cpu",
    }
    assert graph["1"]["inputs"] == {"text": "A lighthouse", "clip": ["0", 1]}
    assert graph["2"]["inputs"] == {"text": "", "clip": ["0", 1]}
    assert inputs(graph, "EmptyFlux2LatentImage") == {
        "width": 1024, "height": 1024, "batch_size": 1,
    }
    assert inputs(graph, "ModelSamplingSD3") == {"model": ["0", 0], "shift": 5.0}
    assert inputs(graph, "SkipLayerGuidanceDiTSimple") == {
        "model": ["4", 0], "double_layers": "", "single_layers": "22",
        "start_percent": 0.0, "end_percent": 1.0,
    }
    assert inputs(graph, "KSampler") == {
        "model": ["5", 0], "positive": ["1", 0], "negative": ["2", 0],
        "latent_image": ["3", 0], "seed": 0, "steps": 50, "cfg": 3.0,
        "sampler_name": "euler", "scheduler": "sgm_uniform", "denoise": 1.0,
    }
    assert inputs(graph, "VAEDecode") == {"samples": ["6", 0], "vae": ["0", 2]}
    assert inputs(graph, "SaveImage") == {"images": ["7", 0], "filename_prefix": "moss-image"}


def test_all_overrides_are_forwarded():
    graph = moss_image_text_to_image(
        "fixture", model_path="/server/model", width=512, height=768, steps=12,
        cfg=2.5, shift=4.0, negative_prompt="blurry", skip_layers="",
        weight_dtype="bf16", text_device="default", batch_size=2, seed=17,
        filename_prefix="custom",
    )
    assert inputs(graph, "MOSSImageHFLoader") == {
        "model_path": "/server/model", "weight_dtype": "bf16", "text_device": "default",
    }
    assert graph["2"]["inputs"]["text"] == "blurry"
    assert inputs(graph, "EmptyFlux2LatentImage") == {"width": 512, "height": 768, "batch_size": 2}
    assert inputs(graph, "ModelSamplingSD3")["shift"] == 4.0
    assert inputs(graph, "SkipLayerGuidanceDiTSimple")["single_layers"] == ""
    assert inputs(graph, "KSampler")["cfg"] == 2.5
    assert inputs(graph, "KSampler")["steps"] == 12
    assert inputs(graph, "KSampler")["seed"] == 17
    assert inputs(graph, "SaveImage")["filename_prefix"] == "custom"


def test_missing_seed_uses_random_uint64(monkeypatch):
    monkeypatch.setattr("creative_comfy_graph.moss_image.secrets.randbits", lambda bits: 123)
    assert inputs(moss_image_text_to_image("fixture"), "KSampler")["seed"] == 123


@pytest.mark.parametrize("options", [
    {"width": 513}, {"height": 0}, {"steps": 0}, {"batch_size": -1},
    {"seed": -1}, {"seed": 2 ** 64}, {"seed": True}, {"weight_dtype": "int4"},
    {"text_device": "cuda"}, {"model_path": " "}, {"cfg": float("nan")},
    {"shift": 0}, {"prompt": ["not", "a string"]},
])
def test_invalid_options_fail_before_submission(options):
    with pytest.raises(ValueError):
        moss_image_text_to_image(**{"prompt": "fixture", **options})


def test_cli_dump_routes_to_image_without_network(legacy_dir):
    script = legacy_dir / "comfy_graph.py"
    environment = dict(os.environ, COMFY_URL_FLUX="http://127.0.0.1:1",
                       COMFY_URL_VIDEO="http://127.0.0.1:2")
    result = subprocess.run(
        [sys.executable, str(script), "dump", "moss-t2i", "--prompt", "fixture",
         "--seed", "0", "--model-path", "/server/model", "--width", "512",
         "--steps", "2", "--negative-prompt", "blurry"],
        env=environment, capture_output=True, text=True, check=True,
    )
    graph = json.loads(result.stdout)
    assert inputs(graph, "KSampler")["seed"] == 0
    assert inputs(graph, "KSampler")["steps"] == 2
    assert inputs(graph, "MOSSImageHFLoader")["model_path"] == "/server/model"
    assert inputs(graph, "EmptyFlux2LatentImage")["width"] == 512
    assert graph["2"]["inputs"]["text"] == "blurry"

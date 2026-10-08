#!/usr/bin/env python3
"""Generate SDPA references and deterministic media for the public Swift APIs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
import wave

import numpy as np
from PIL import Image
from safetensors.torch import load_file, save_file
import torch
from sentence_transformers import SentenceTransformer

MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


def pytorch_checkpoint(source, destination):
    for entry in source.iterdir():
        if entry.is_dir():
            shutil.copytree(entry, destination / entry.name, dirs_exist_ok=True)
        elif entry.suffix in (".json", ".jinja"):
            shutil.copy2(entry, destination / entry.name)
    weights = {}
    for file in sorted(source.glob("*.safetensors")):
        for name, tensor in load_file(str(file)).items():
            if name.endswith("conv.weight") and tensor.ndim == 4:
                tensor = tensor.permute(0, 3, 1, 2).contiguous()
            elif name.endswith("depthwise_conv1d.weight"):
                tensor = tensor.permute(0, 2, 1).contiguous()
            weights[name] = tensor
    if not weights:
        raise ValueError("No MLX safetensors found")
    save_file(weights, str(destination / "model.safetensors"), metadata={"format": "pt"})
    index = destination / "model.safetensors.index.json"
    if index.exists():
        index.unlink()


def generate(args, model_path):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:240, :320]
    pixels = np.stack([(x * 3 + y) % 256, (x + y * 5) % 256, (x * 7 + y * 2) % 256], axis=-1)
    image = Image.fromarray(pixels.astype(np.uint8))
    image.save(output / "image.png")
    frames = []
    for index in range(5):
        frame = Image.fromarray(np.roll(pixels, index * 30, axis=1).astype(np.uint8))
        frame.save(output / f"frame{index}.png")
        frames.append(frame)
    time = np.arange(16000 * 3, dtype=np.float64) / 16000
    samples = (0.2 * np.sin(2 * np.pi * 440 * time) + 0.1 * np.sin(2 * np.pi * 880 * time))
    pcm = np.rint(samples * 32767).astype("<i2")
    with wave.open(str(output / "audio.wav"), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)
        audio_file.writeframes(pcm.tobytes())
    audio = pcm.astype(np.float32) / 32768
    revision = args.revision if not Path(model_path).is_dir() else None
    model = SentenceTransformer(model_path, revision=revision, device="cpu",
        model_kwargs={"torch_dtype": torch.float32, "attn_implementation": "sdpa"})
    captured = {}
    auto_model = model[0].auto_model
    original = auto_model.forward
    def forward(*positional, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return original(*positional, **kwargs)
    auto_model.forward = forward
    media = {"image.png": image, "frames": frames, "audio.wav": audio}
    paragraph = "Auroras form when charged solar particles collide with gases in the upper atmosphere. "
    cases = [
        ("query", "searchQuery", "SearchQuery", [["text", "What causes the northern lights?"]]),
        ("document", "document", "Document", [["text", "The northern lights are caused by charged particles from the sun."]]),
        ("long-text", "document", "Document", [["text", paragraph * 180]]),
        ("image", "none", None, [["image", "image.png"]]),
        ("audio", "none", None, [["audio", "audio.wav"]]),
        ("video", "none", None, [["video", "frames"]]),
        ("interleaved", "document", "Document", [["text", "Compare these patterns: "], ["image", "image.png"], ["video", "frames"], ["audio", "audio.wav"]]),
        ("image-before-text", "searchQuery", "SearchQuery", [["image", "image.png"], ["text", " Colorful patterns"]]),
    ]
    records = []
    with torch.inference_mode():
        for name, task, prompt, parts in cases:
            sample = {kind: value if kind == "text" else media[value] for kind, value in parts}
            vector = model.encode(sample, prompt_name=prompt, normalize_embeddings=True)
            tokens = captured["input_ids"][0].tolist()
            if len(tokens) > 8192:
                raise ValueError(f"{name} exceeds the context window")
            records.append({"name": name, "task": task, "parts": parts, "tokens": tokens,
                "embedding": vector.astype(float).tolist()})
            print(f"{name}: {len(tokens)} tokens", flush=True)
    versions = {name: importlib.metadata.version(name) for name in
        ["torch", "torchvision", "transformers", "sentence-transformers", "numpy", "pillow", "safetensors"]}
    hashes = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in output.iterdir()
        if file.suffix in (".png", ".wav")}
    source = str(args.mlx_model.resolve()) if args.mlx_model else args.model
    source_revision = args.revision if not Path(source).is_dir() else None
    manifest = {"model": source, "revision": source_revision, "attention": "sdpa", "dtype": "float32",
        "versions": versions, "media_sha256": hashes, "cases": records}
    if args.mlx_model:
        manifest["checkpoint_sha256"] = {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(args.mlx_model.glob("*.safetensors"))}
    (output / "reference.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/embeddinggemma-2")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--mlx-model", type=Path, help="Convert a local MLX-layout checkpoint for parity")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mlx_model:
        with tempfile.TemporaryDirectory(prefix="embeddinggemma2-reference-") as directory:
            pytorch_checkpoint(args.mlx_model, Path(directory))
            generate(args, directory)
    else:
        generate(args, args.model)


if __name__ == "__main__":
    main()

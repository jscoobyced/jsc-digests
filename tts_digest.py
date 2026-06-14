#!/usr/bin/env python3
"""Convert generated digest markdown files to audio using Spark-TTS-0.5B.

Walks <output_dir>/<date>/<source>/*.md and writes a sibling .wav next to
each markdown file. The index.md and any pre-existing .wav are skipped.

Setup (one-time):
  1. Clone Spark-TTS:
       git clone https://github.com/SparkAudio/Spark-TTS.git ./Spark-TTS
  2. Download the model weights into the cloned repo:
       cd ./Spark-TTS && mkdir -p pretrained_models
       huggingface-cli download SparkAudio/Spark-TTS-0.5B \
           --local-dir pretrained_models/Spark-TTS-0.5B
  3. Install Spark-TTS deps + this script's deps:
       pip install -r ./Spark-TTS/requirements.txt
       pip install -r requirements-tts.txt
  4. Tell this script where Spark-TTS lives:
       export SPARK_TTS_PATH=./Spark-TTS
       export SPARK_TTS_MODEL_DIR=$SPARK_TTS_PATH/pretrained_models/Spark-TTS-0.5B

Usage:
  python tts_digest.py                    # process today's digest
  python tts_digest.py 2026-06-11         # process a specific date
  python tts_digest.py --root ./digests/2026-06-11
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_output_root() -> Path:
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    return Path(cfg["output"]["dir"]).expanduser().resolve()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def import_spark_tts():
    spark_path = os.environ.get("SPARK_TTS_PATH")
    if not spark_path:
        print("ERROR: set SPARK_TTS_PATH to your cloned Spark-TTS repo", file=sys.stderr)
        sys.exit(2)
    spark_path = str(Path(spark_path).expanduser().resolve())
    if spark_path not in sys.path:
        sys.path.insert(0, spark_path)
    # The class lives in cli/SparkTTS.py; sparktts/ is a sibling package it imports from.
    try:
        from cli.SparkTTS import SparkTTS  # type: ignore
    except ImportError as e:
        print(
            f"ERROR: could not import SparkTTS from {spark_path}: {e}\n"
            f"Expected file: {spark_path}/cli/SparkTTS.py "
            f"(make sure SPARK_TTS_PATH points at the cloned Spark-TTS repo root)",
            file=sys.stderr,
        )
        sys.exit(2)
    return SparkTTS


# Markdown → plain text
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HEADER_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_BULLET_RE = re.compile(r"^[\-\*]\s+", re.MULTILINE)
_EMPH_RE = re.compile(r"[*_`]+")
_BLANK_RE = re.compile(r"\n{3,}")


def markdown_to_speech_text(md: str) -> str:
    text = _LINK_RE.sub(r"\1", md)
    text = _HEADER_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _EMPH_RE.sub("", text)
    text = _BLANK_RE.sub("\n\n", text).strip()
    return text


def find_markdown_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for md in sorted(root.rglob("*.md")):
        if md.name == "index.md":
            continue
        files.append(md)
    return files


def synthesize(model, text: str, *, device: torch.device):
    """Call SparkTTS.inference for voice-cloning-free generation."""
    with torch.no_grad():
        wav = model.inference(
            text=text,
            prompt_speech_path=None,
            prompt_text=None,
            gender="female",
            pitch="moderate",
            speed="moderate",
        )
    return wav


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert digest markdown to audio with Spark-TTS-0.5B")
    parser.add_argument("date", nargs="?", default=None, help="Digest date in YYYY-MM-DD (default: today)")
    parser.add_argument("--root", default=None, help="Override the digest date folder explicitly")
    parser.add_argument("--force", action="store_true", help="Re-generate audio even if .wav already exists")
    args = parser.parse_args()

    if args.root:
        date_root = Path(args.root).expanduser().resolve()
    else:
        d = args.date or date.today().isoformat()
        date_root = load_output_root() / d

    md_files = find_markdown_files(date_root)
    if not md_files:
        print(f"No markdown files found under {date_root}", file=sys.stderr)
        return 1

    model_dir = os.environ.get("SPARK_TTS_MODEL_DIR")
    if not model_dir:
        print("ERROR: set SPARK_TTS_MODEL_DIR to the Spark-TTS-0.5B model directory", file=sys.stderr)
        return 2

    SparkTTS = import_spark_tts()
    device = pick_device()
    print(f"Loading Spark-TTS-0.5B from {model_dir} on {device}", file=sys.stderr)
    model = SparkTTS(model_dir, device)

    n_done = 0
    for md_path in md_files:
        wav_path = md_path.with_suffix(".wav")
        if wav_path.exists() and not args.force:
            print(f"skip (exists): {wav_path}")
            continue
        text = markdown_to_speech_text(md_path.read_text())
        if not text.strip():
            print(f"skip (empty): {md_path}")
            continue
        print(f"synthesizing: {md_path.name}")
        try:
            wav = synthesize(model, text, device=device)
        except Exception as e:
            print(f"  failed: {e}", file=sys.stderr)
            continue
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()
        wav = np.asarray(wav).squeeze()
        sf.write(str(wav_path), wav, model.sample_rate)
        print(f"  wrote {wav_path}")
        n_done += 1

    print(f"Done. Generated {n_done} audio file(s) under {date_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

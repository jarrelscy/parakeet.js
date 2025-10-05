#!/usr/bin/env python3
"""Regenerate audiosave transcripts with the onnx_asr reference decoder."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

try:
    import onnx_asr
except ModuleNotFoundError as exc:  # pragma: no cover - handled in CI setup
    missing = exc.name or "onnx_asr"
    raise SystemExit(f"{missing} must be installed to refresh transcripts") from exc

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError as exc:  # pragma: no cover - handled in CI setup
    missing = exc.name or "huggingface_hub"
    raise SystemExit(f"{missing} must be installed to refresh transcripts") from exc


def load_audio(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1:
            raise RuntimeError(f"{path} must be mono")
        if wav.getframerate() != 16000:
            raise RuntimeError(f"{path} must be 16000Hz")
        if wav.getsampwidth() != 2:
            raise RuntimeError(f"{path} must be 16-bit PCM")
        frames = wav.readframes(wav.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    return samples


def _stage_model_directory(model_dir: str | None) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    if model_dir:
        source = Path(model_dir)
        if not source.exists():
            raise SystemExit(f"Model directory {source} does not exist")
    else:
        source_path = snapshot_download(
            "jarrelscy/parakeet-tdt-0.6b-v2-onnx",
            allow_patterns=[
                "config.json",
                "encoder-model.onnx",
                "encoder-model.onnx.data",
                "decoder_joint-model.onnx",
                "vocab.txt",
                "nemo128.onnx",
            ],
        )
        source = Path(source_path)

    temp_dir = tempfile.TemporaryDirectory()
    staged = Path(temp_dir.name)

    def copy_if_exists(name: str, *, rename: str | None = None) -> Path | None:
        src = source / name
        if not src.exists():
            return None
        dest = staged / (rename or name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest

    # Always copy encoder weights (only fp32 is available today).
    encoder_path = copy_if_exists("encoder-model.onnx")
    if encoder_path is None:
        raise SystemExit(f"Missing encoder-model.onnx in {source}")

    copy_if_exists("encoder-model.onnx.data")

    # Use the fp32 decoder to match the JavaScript configuration.
    decoder_copied = copy_if_exists("decoder_joint-model.onnx")
    if decoder_copied is None:
        raise SystemExit(f"Missing decoder weights in {source}")

    if copy_if_exists("vocab.txt") is None:
        raise SystemExit(f"Missing vocab.txt in {source}")

    copy_if_exists("nemo128.onnx")
    copy_if_exists("config.json")

    return temp_dir, staged


def regenerate_transcripts(audio_dir: Path, model_dir: str | None) -> None:
    temp_dir, staged_dir = _stage_model_directory(model_dir)
    try:
        model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", str(staged_dir))

        wav_files = sorted(p for p in audio_dir.iterdir() if p.suffix == ".wav")
        if not wav_files:
            raise SystemExit(f"No .wav files found in {audio_dir}")

        for wav_path in wav_files:
            samples = load_audio(wav_path)
            text = model.recognize(samples, sample_rate=16000)
            out_path = wav_path.with_suffix(".txt")
            out_path.write_text(text, encoding="utf-8")
            print(f"{wav_path.name}:\n{text}\n")
    finally:
        temp_dir.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-dir",
        default=Path(__file__).resolve().parents[1] / "audiosave",
        type=Path,
        help="Directory containing .wav files to transcribe",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("PARAKEET_LOCAL_MODEL_DIR"),
        help="Optional directory with cached Parakeet ONNX weights",
    )
    args = parser.parse_args(argv)

    audio_dir = args.audio_dir
    if not audio_dir.exists():
        raise SystemExit(f"Audio directory {audio_dir} does not exist")

    regenerate_transcripts(audio_dir, args.model_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

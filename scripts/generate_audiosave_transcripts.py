#!/usr/bin/env python3
"""Regenerate audiosave transcripts with the onnx_asr reference decoder."""

from __future__ import annotations

import argparse
import os
import sys
import wave
from pathlib import Path

import numpy as np

try:
    import onnx_asr
except ModuleNotFoundError as exc:  # pragma: no cover - handled in CI setup
    raise SystemExit("onnx_asr must be installed to refresh transcripts") from exc


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


def regenerate_transcripts(audio_dir: Path, model_dir: str | None) -> None:
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", model_dir)

    wav_files = sorted(p for p in audio_dir.iterdir() if p.suffix == ".wav")
    if not wav_files:
        raise SystemExit(f"No .wav files found in {audio_dir}")

    for wav_path in wav_files:
        samples = load_audio(wav_path)
        text = model.recognize(samples, sample_rate=16000)
        out_path = wav_path.with_suffix(".txt")
        out_path.write_text(text, encoding="utf-8")


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

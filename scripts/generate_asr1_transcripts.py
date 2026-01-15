#!/usr/bin/env python3
"""Regenerate audiosave transcripts with the jarrelscy/asr1 ONNX model."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
import wave
from pathlib import Path

import numpy as np
import re

try:
    import onnxruntime as ort
except ModuleNotFoundError as exc:  # pragma: no cover - handled in CI setup
    missing = exc.name or "onnxruntime"
    raise SystemExit(f"{missing} must be installed to refresh transcripts") from exc


LOG_FLOOR = 1e-10


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


def hz_to_mel(hz: float) -> float:
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel: float) -> float:
    return 700 * (10 ** (mel / 2595) - 1)


def create_mel_filterbank(n_mels: int, n_fft: int, sample_rate: int) -> np.ndarray:
    fft_bins = n_fft // 2 + 1
    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_mels, fft_bins), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = bin_points[m], bin_points[m + 1], bin_points[m + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for k in range(left, center):
            if 0 <= k < fft_bins:
                filters[m, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            if 0 <= k < fft_bins:
                filters[m, k] = (right - k) / max(right - center, 1)
    return filters


def log_mel_spectrogram(audio: np.ndarray, sample_rate: int, n_fft: int, win_length: int, hop_length: int, n_mels: int) -> np.ndarray:
    if audio.shape[0] <= win_length:
        frames = 1
    else:
        frames = (audio.shape[0] - win_length) // hop_length + 1
        remainder = (audio.shape[0] - win_length) % hop_length
        if remainder:
            frames += 1
    needed = (frames - 1) * hop_length + win_length
    if audio.shape[0] < needed:
        pad_width = needed - audio.shape[0]
        audio = np.pad(audio, (0, pad_width))

    window = np.hanning(win_length).astype(np.float32)
    filters = create_mel_filterbank(n_mels, n_fft, sample_rate)

    features = np.zeros((frames, n_mels), dtype=np.float32)
    for idx in range(frames):
        start = idx * hop_length
        frame = audio[start:start + win_length] * window
        spectrum = np.fft.rfft(frame, n=n_fft)
        power = (np.abs(spectrum) ** 2).astype(np.float32)
        mel = np.dot(filters, power)
        features[idx] = np.log(np.maximum(mel, LOG_FLOOR))
    return features


def download_file(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as resp, dest.open("wb") as handle:
        handle.write(resp.read())


def prepare_model(tmp_dir: Path) -> tuple[Path, Path, Path]:
    model_path = tmp_dir / "model.onnx"
    data_path = tmp_dir / "model.onnx.data"
    vocab_path = tmp_dir / "tokenizer.json"

    base = "https://huggingface.co/jarrelscy/asr1/resolve/main"
    download_file(f"{base}/model.onnx", model_path)
    download_file(f"{base}/model.onnx.data", data_path)
    download_file(f"{base}/tokenizer.json", vocab_path)
    return model_path, data_path, vocab_path


def load_vocab(path: Path) -> list[str]:
    data = path.read_text(encoding="utf-8")
    if data.strip().startswith("{"):
        payload = json.loads(data)
        vocab = payload.get("model", {}).get("vocab", [])
        if isinstance(vocab, list):
            return [entry[0] for entry in vocab]
        if isinstance(vocab, dict):
            id2token = []
            for token, idx in vocab.items():
                if idx >= len(id2token):
                    id2token.extend([""] * (idx - len(id2token) + 1))
                id2token[idx] = token
            return id2token
    id2token = []
    for line in data.splitlines():
        if not line.strip():
            continue
        token, idx = line.split()
        id2token.extend([""] * (int(idx) - len(id2token) + 1))
        id2token[int(idx)] = token
    return id2token


def decode(ids: list[int], id2token: list[str]) -> str:
    pieces = []
    skip_tokens = {"<epsilon>", "<s>", "</s>", "<unk>"}
    for idx in ids:
        if idx < 0 or idx >= len(id2token):
            continue
        token = id2token[idx]
        if token in skip_tokens:
            continue
        pieces.append(token.replace("▁", " "))
    raw = "".join(pieces)
    if not raw:
        return ""
    return re.sub(r"(^\s|\s\B|(\s)\b)", lambda m: " " if m.group(2) else "", raw)


def regenerate_transcripts(audio_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path, _data_path, vocab_path = prepare_model(Path(temp_dir))
        id2token = load_vocab(vocab_path)

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        wav_files = sorted(p for p in audio_dir.iterdir() if p.suffix == ".wav")
        if not wav_files:
            raise SystemExit(f"No .wav files found in {audio_dir}")

        for wav_path in wav_files:
            samples = load_audio(wav_path)
            features = log_mel_spectrogram(samples, 16000, 512, 400, 160, 128)
            feats = features[np.newaxis, :, :].astype(np.float32)
            logits = session.run(None, {input_name: feats})[0]
            argmax = np.argmax(logits, axis=-1)[0]
            ids = []
            prev = None
            for idx in argmax:
                if idx == 0:
                    prev = idx
                    continue
                if idx != prev:
                    ids.append(int(idx))
                prev = idx
            text = decode(ids, id2token)
            out_path = wav_path.with_suffix(".asr1.txt")
            out_path.write_text(text.strip(), encoding="utf-8")
            print(f"{wav_path.name}:\n{text}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-dir",
        default=Path(__file__).resolve().parents[1] / "audiosave",
        type=Path,
        help="Directory containing .wav files to transcribe",
    )
    args = parser.parse_args(argv)

    audio_dir = args.audio_dir
    if not audio_dir.exists():
        raise SystemExit(f"Audio directory {audio_dir} does not exist")

    regenerate_transcripts(audio_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

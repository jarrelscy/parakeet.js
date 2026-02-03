#!/usr/bin/env python3
"""Regenerate audiosave transcripts with the jarrelscy/asr1 model.

Uses ONNX Runtime directly with LasrFeatureExtractor preprocessing to match
what the JavaScript implementation does.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import librosa
import numpy as np

try:
    import onnxruntime as ort
except ModuleNotFoundError as exc:
    missing = exc.name or "onnxruntime"
    raise SystemExit(f"{missing} must be installed to refresh transcripts") from exc


# LasrFeatureExtractor parameters (matching HuggingFace)
LOG_FLOOR = 1e-5
LOWER_EDGE_HERTZ = 125.0
UPPER_EDGE_HERTZ = 7500.0


def hz_to_mel_kaldi(hz: float) -> float:
    """Kaldi mel scale conversion."""
    return 1127.0 * np.log(1.0 + hz / 700.0)


def create_mel_filterbank_hf(n_mels: int, n_fft: int, sample_rate: int,
                              lower_edge_hz: float = 125.0, upper_edge_hz: float = 7500.0) -> np.ndarray:
    """Create mel filterbank matching HuggingFace's linear_to_mel_weight_matrix."""
    num_spectrogram_bins = n_fft // 2 + 1
    bands_to_zero = 1  # Excludes DC bin
    nyquist = sample_rate / 2.0

    # Linear frequencies (excluding DC)
    linear_freqs = np.array([(i / (num_spectrogram_bins - 1)) * nyquist
                             for i in range(bands_to_zero, num_spectrogram_bins)])

    # Convert linear frequencies to mel (Kaldi scale)
    spectrogram_bins_mel = hz_to_mel_kaldi(linear_freqs)

    # Mel band edges
    lower_mel = hz_to_mel_kaldi(lower_edge_hz)
    upper_mel = hz_to_mel_kaldi(upper_edge_hz)
    edges = np.linspace(lower_mel, upper_mel, n_mels + 2)

    # Create filterbank with shape [num_spectrogram_bins, n_mels]
    filters = np.zeros((num_spectrogram_bins, n_mels), dtype=np.float64)

    for m in range(n_mels):
        lower_edge_mel = edges[m]
        center_mel = edges[m + 1]
        upper_edge_mel = edges[m + 2]

        for i, bin_mel in enumerate(spectrogram_bins_mel):
            lower_slope = (bin_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
            upper_slope = (upper_edge_mel - bin_mel) / (upper_edge_mel - center_mel)
            weight = max(0.0, min(lower_slope, upper_slope))
            # Account for bands_to_zero offset
            filters[i + bands_to_zero, m] = weight

    return filters


def extract_features(audio: np.ndarray, sample_rate: int = 16000, n_fft: int = 512,
                     win_length: int = 400, hop_length: int = 160, n_mels: int = 128) -> np.ndarray:
    """Extract log-mel spectrogram features matching HuggingFace LasrFeatureExtractor."""
    # Calculate number of frames using unfold logic
    num_frames = (len(audio) - win_length) // hop_length + 1
    if num_frames <= 0:
        raise ValueError(f"Audio too short: {len(audio)} samples, need at least {win_length}")

    # Hann window (periodic=False, matching PyTorch)
    window = np.hanning(win_length).astype(np.float64)

    # Mel filterbank
    mel_filters = create_mel_filterbank_hf(n_mels, n_fft, sample_rate,
                                           LOWER_EDGE_HERTZ, UPPER_EDGE_HERTZ)

    features = np.zeros((num_frames, n_mels), dtype=np.float32)

    for frame in range(num_frames):
        offset = frame * hop_length
        windowed = audio[offset:offset + win_length] * window

        # RFFT
        spectrum = np.fft.rfft(windowed, n=n_fft)
        power_spec = np.abs(spectrum) ** 2

        # Apply mel filterbank
        mel_spec = power_spec @ mel_filters

        # Log with floor
        features[frame] = np.log(np.maximum(mel_spec, LOG_FLOOR))

    return features


def download_file(url: str, dest: Path) -> None:
    """Download file from URL to destination."""
    with urllib.request.urlopen(url) as resp, dest.open("wb") as handle:
        handle.write(resp.read())


def prepare_model(tmp_dir: Path) -> tuple[Path, Path]:
    """Download model files from HuggingFace."""
    model_path = tmp_dir / "model.onnx"
    data_path = tmp_dir / "model.onnx.data"
    vocab_path = tmp_dir / "vocab.json"

    base = "https://huggingface.co/jarrelscy/asr1/resolve/main"
    print("Downloading model files...")
    download_file(f"{base}/model.onnx", model_path)
    download_file(f"{base}/model.onnx.data", data_path)
    download_file(f"{base}/vocab.json", vocab_path)
    return model_path, vocab_path


def load_vocab(path: Path) -> list[str]:
    """Load vocabulary from vocab.json."""
    data = path.read_text(encoding="utf-8")
    vocab = json.loads(data)
    # vocab.json is token->id mapping, invert it
    id2token = [""] * (max(vocab.values()) + 1)
    for token, idx in vocab.items():
        id2token[idx] = token
    return id2token


def ctc_decode(logits: np.ndarray, id2token: list[str], blank_id: int = 0) -> str:
    """CTC decoding: argmax, collapse duplicates, remove blanks, decode tokens."""
    # Argmax
    argmax = np.argmax(logits, axis=-1)[0]  # Remove batch dim

    # Collapse consecutive duplicates and remove blanks
    ids = []
    prev = None
    for idx in argmax:
        if idx == blank_id:
            prev = idx
            continue
        if idx != prev:
            ids.append(int(idx))
        prev = idx

    # Decode tokens - matches HuggingFace tokenizer.decode behavior
    skip_tokens = {"<epsilon>", "<s>", "</s>", "<unk>"}
    pieces = []
    for idx in ids:
        if idx < 0 or idx >= len(id2token):
            continue
        token = id2token[idx]
        if token in skip_tokens:
            continue
        pieces.append(token.replace("▁", " "))

    # Return raw joined output without cleanup - this is the ground truth
    return "".join(pieces)


def regenerate_transcripts(audio_dir: Path) -> None:
    """Regenerate transcripts for all WAV files in the given directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path, vocab_path = prepare_model(Path(temp_dir))
        id2token = load_vocab(vocab_path)

        print("Loading ONNX model...")
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        wav_files = sorted(p for p in audio_dir.iterdir() if p.suffix == ".wav")
        if not wav_files:
            raise SystemExit(f"No .wav files found in {audio_dir}")

        for wav_path in wav_files:
            # Load audio at 16kHz
            audio, sr = librosa.load(str(wav_path), sr=16000)

            # Extract features using LasrFeatureExtractor preprocessing
            features = extract_features(audio)

            # Add batch dimension [1, T, n_mels]
            features = features[np.newaxis, :, :].astype(np.float32)

            # Run inference
            logits = session.run(None, {input_name: features})[0]

            # CTC decode
            text = ctc_decode(logits, id2token, blank_id=0)

            # Write output
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

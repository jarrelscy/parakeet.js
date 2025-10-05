import test from 'node:test';
import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawn } from 'node:child_process';

import './helpers/setup-node-env.js';

import { getParakeetModel } from '../src/hub.js';
import { ParakeetModel } from '../src/parakeet.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const projectRoot = path.resolve(__dirname, '..');
const audioDir = path.join(projectRoot, 'audiosave');

function isNetworkError(err) {
  if (!err) return false;
  if (err.code === 'ENETUNREACH' || err.code === 'EAI_AGAIN') return true;
  if (err.cause && isNetworkError(err.cause)) return true;
  if (typeof err.errors === 'object' && err.errors) {
    for (const nested of err.errors) {
      if (isNetworkError(nested)) return true;
    }
  }
  const msg = err.message ? String(err.message) : '';
  return msg.includes('ENETUNREACH') || msg.includes('fetch failed');
}

async function loadWavFloat32(filePath) {
  const buffer = await fs.readFile(filePath);
  if (buffer.length < 44) {
    throw new Error('Invalid WAV file – too short');
  }

  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (riff !== 'RIFF') {
    throw new Error('Invalid WAV file – missing RIFF header');
  }
  const wave = String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11));
  if (wave !== 'WAVE') {
    throw new Error('Invalid WAV file – missing WAVE signature');
  }

  let offset = 12;
  let fmtChunkFound = false;
  let audioFormat = 0;
  let numChannels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;
  let dataOffset = -1;
  let dataSize = 0;

  while (offset + 8 <= buffer.length) {
    const chunkId = String.fromCharCode(
      view.getUint8(offset),
      view.getUint8(offset + 1),
      view.getUint8(offset + 2),
      view.getUint8(offset + 3)
    );
    const chunkSize = view.getUint32(offset + 4, true);
    offset += 8;

    if (chunkId === 'fmt ') {
      fmtChunkFound = true;
      audioFormat = view.getUint16(offset, true);
      numChannels = view.getUint16(offset + 2, true);
      sampleRate = view.getUint32(offset + 4, true);
      bitsPerSample = view.getUint16(offset + 14, true);
    } else if (chunkId === 'data') {
      dataOffset = offset;
      dataSize = chunkSize;
      break;
    }

    offset += chunkSize;
  }

  if (!fmtChunkFound || dataOffset < 0) {
    throw new Error('Invalid WAV file – missing fmt/data chunks');
  }

  if (audioFormat !== 1) {
    throw new Error(`Unsupported WAV audio format ${audioFormat}; expected PCM`);
  }

  if (bitsPerSample !== 16) {
    throw new Error(`Unsupported WAV bit depth ${bitsPerSample}; expected 16-bit PCM`);
  }

  if (numChannels !== 1) {
    throw new Error(`Unsupported WAV channel count ${numChannels}; expected mono`);
  }

  if (sampleRate !== 16000) {
    throw new Error(`Unexpected sample rate ${sampleRate}; expected 16000 Hz`);
  }

  const pcmData = buffer.subarray(dataOffset, dataOffset + dataSize);
  const samples = new Int16Array(pcmData.buffer, pcmData.byteOffset, pcmData.byteLength / 2);
  const floatData = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    floatData[i] = samples[i] / 32768;
  }
  return floatData;
}

let modelPromise;

async function getModel() {
  if (!modelPromise) {
    const wasmDir = path.resolve(projectRoot, 'node_modules/onnxruntime-web/dist');
    const wasmDirUrl = pathToFileURL(wasmDir).href;
    const wasmUrl = wasmDirUrl.endsWith('/') ? wasmDirUrl : `${wasmDirUrl}/`;

    const localRepo = process.env.PARAKEET_LOCAL_MODEL_DIR;
    let modelConfig;

    if (localRepo) {
      const base = path.resolve(projectRoot, localRepo);
      async function ensure(name) {
        const filePath = path.join(base, name);
        try {
          await fs.access(filePath);
          return filePath;
        } catch (err) {
          if (err.code === 'ENOENT') return null;
          throw err;
        }
      }

      const encoderPath = await ensure('encoder-model.onnx');
      const encoderDataPath = await ensure('encoder-model.onnx.data');
      const decoderPath = await ensure('decoder_joint-model.onnx');
      const vocabPath = await ensure('vocab.txt');
      const preprocPath = await ensure('nemo128.onnx');

      modelConfig = {
        urls: {
          encoderUrl: encoderPath,
          encoderDataUrl: encoderDataPath,
          decoderUrl: decoderPath,
          tokenizerUrl: vocabPath ? pathToFileURL(vocabPath).href : null,
          preprocessorUrl: preprocPath,
        },
        filenames: {
          encoder: 'encoder-model.onnx',
          decoder: 'decoder_joint-model.onnx',
        },
      };

      if (!encoderPath || !decoderPath || !vocabPath || !preprocPath) {
        throw new Error(`PARAKEET_LOCAL_MODEL_DIR is missing required files in ${base}`);
      }
    } else {
      modelConfig = await getParakeetModel('jarrelscy/parakeet-tdt-0.6b-v2-onnx', {
        backend: 'wasm',
        encoderQuant: 'fp32',
        decoderQuant: 'fp32',
      });
    }

    modelPromise = ParakeetModel.fromUrls({
      ...modelConfig.urls,
      filenames: modelConfig.filenames,
      backend: 'wasm',
      wasmPaths: wasmUrl,
      decoderQuant: 'fp32',
      encoderQuant: 'fp32',
    });
  }
  return modelPromise;
}

class SkipTestError extends Error {
  constructor(message) {
    super(message);
    this.name = 'SkipTestError';
  }
}

let pythonReferencePromise;

async function ensurePythonTranscripts() {
  if (process.env.PARAKEET_SKIP_PYTHON === '1') {
    return;
  }

  if (!pythonReferencePromise) {
    pythonReferencePromise = (async () => {
      const pythonBin = process.env.PYTHON || process.env.PYTHON_BIN || 'python3';
      const localRepo = process.env.PARAKEET_LOCAL_MODEL_DIR
        ? path.resolve(projectRoot, process.env.PARAKEET_LOCAL_MODEL_DIR)
        : '';

      const script = `import os
import sys
import wave

EXIT_SKIP = 75

audio_dir = sys.argv[1]
model_dir = sys.argv[2] or None

try:
    import numpy as np
except ModuleNotFoundError:
    sys.stderr.write('python reference skipped: numpy module not found\\n')
    sys.exit(EXIT_SKIP)

try:
    import onnx_asr
except ModuleNotFoundError:
    sys.stderr.write('python reference skipped: onnx_asr module not found\\n')
    sys.exit(EXIT_SKIP)


def load_audio(path):
    with wave.open(path, 'rb') as wav:
        if wav.getnchannels() != 1:
            raise RuntimeError(f"{path} must be mono")
        if wav.getframerate() != 16000:
            raise RuntimeError(f"{path} must be 16000Hz")
        if wav.getsampwidth() != 2:
            raise RuntimeError(f"{path} must be 16-bit PCM")
        frames = wav.readframes(wav.getnframes())
    samples = np.frombuffer(frames, dtype='<i2').astype('float32') / 32768.0
    return samples


model = onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v2', model_dir)

for entry in sorted(os.listdir(audio_dir)):
    if not entry.endswith('.wav'):
        continue
    wav_path = os.path.join(audio_dir, entry)
    samples = load_audio(wav_path)
    text = model.recognize(samples, sample_rate=16000)
    out_path = os.path.join(audio_dir, entry[:-4] + '.txt')
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(text)
sys.exit(0)
`;

      const args = ['-', audioDir, localRepo];

      let stdout = '';
      let stderr = '';

      const runResult = await new Promise((resolve, reject) => {
        const child = spawn(pythonBin, args, {
          stdio: ['pipe', 'pipe', 'pipe'],
          env: process.env,
        });

        child.stdin.end(script);

        child.stdout.on('data', (chunk) => {
          stdout += chunk.toString();
        });

        child.stderr.on('data', (chunk) => {
          stderr += chunk.toString();
        });

        child.on('error', (err) => {
          reject(err);
        });

        child.on('close', (code) => {
          resolve({ code });
        });
      }).catch((err) => {
        if (err && err.code === 'ENOENT') {
          throw new SkipTestError(`Python interpreter not found: ${pythonBin}`);
        }
        throw err;
      });

      if (runResult.code === 0) {
        if (stdout.trim().length > 0) {
          process.stdout.write(stdout);
        }
        if (stderr.trim().length > 0) {
          process.stderr.write(stderr);
        }
        return;
      }

      if (runResult.code === 75) {
        throw new SkipTestError(stderr.trim() || 'Python reference generation skipped');
      }

      const details = [
        `exit code ${runResult.code}`,
        stderr.trim() ? `stderr: ${stderr.trim()}` : null,
        stdout.trim() ? `stdout: ${stdout.trim()}` : null,
      ]
        .filter(Boolean)
        .join('\n');

      throw new Error(`Python reference generation failed\n${details}`);
    })();
  }

  return pythonReferencePromise;
}

async function collectAudioPairs() {
  await ensurePythonTranscripts();
  const files = await fs.readdir(audioDir);
  const wavFiles = files.filter((f) => f.endsWith('.wav')).sort();
  const pairs = [];
  for (const wavFile of wavFiles) {
    const base = wavFile.replace(/\.wav$/, '');
    const txtFile = `${base}.txt`;
    const txtPath = path.join(audioDir, txtFile);
    try {
      const expected = await fs.readFile(txtPath, 'utf8');
      pairs.push({
        name: base,
        wavPath: path.join(audioDir, wavFile),
        expected: expected.trim(),
      });
    } catch (err) {
      throw new Error(`Missing transcript file for ${wavFile}: ${err.message}`);
    }
  }
  return pairs;
}

test('parakeet wasm transcription matches reference texts', { timeout: 600_000 }, async (t) => {
  let pairs;
  try {
    pairs = await collectAudioPairs();
  } catch (err) {
    if (err instanceof SkipTestError) {
      t.skip(err.message);
      return;
    }
    throw err;
  }

  let model;
  try {
    model = await getModel();
  } catch (err) {
    if (isNetworkError(err)) {
      t.skip('Network unavailable for HuggingFace model download');
      return;
    }
    throw err;
  }

  for (const pair of pairs) {
    await t.test(`transcribes ${pair.name}`, async () => {
      const audio = await loadWavFloat32(pair.wavPath);
      const result = await model.transcribe(audio, 16000, { returnTimestamps: false, returnConfidences: false });
      assert.equal(result.utterance_text.trim(), pair.expected);
    });
  }
});

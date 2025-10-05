import test from 'node:test';
import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

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

async function collectAudioPairs() {
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

  const pairs = await collectAudioPairs();

  for (const pair of pairs) {
    await t.test(`transcribes ${pair.name}`, async () => {
      const audio = await loadWavFloat32(pair.wavPath);
      const result = await model.transcribe(audio, 16000, { returnTimestamps: false, returnConfidences: false });
      assert.equal(result.utterance_text.trim(), pair.expected);
    });
  }
});

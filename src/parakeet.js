import { initOrt } from './backend.js';
import { ParakeetTokenizer } from './tokenizer.js';
import { OnnxPreprocessor } from './preprocessor.js';

function float16ToFloat32Array(src) {
  const dst = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const h = src[i];
    const sign = (h & 0x8000) ? -1 : 1;
    const exponent = (h & 0x7C00) >> 10;
    const fraction = h & 0x03FF;

    if (exponent === 0) {
      if (fraction === 0) {
        dst[i] = sign === -1 ? -0 : 0;
      } else {
        dst[i] = sign * Math.pow(2, -14) * (fraction / 1024);
      }
    } else if (exponent === 0x1F) {
      dst[i] = fraction ? NaN : (sign === -1 ? -Infinity : Infinity);
    } else {
      dst[i] = sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
    }
  }
  return dst;
}

function tensorDataToFloat32(tensor) {
  if (!tensor) return null;
  const data = tensor.data;
  if (!data) return null;

  if (tensor.type === 'float32' || data instanceof Float32Array) {
    return data;
  }

  if (tensor.type === 'float64' || data instanceof Float64Array) {
    return Float32Array.from(data);
  }

  if (tensor.type === 'float16' || data instanceof Uint16Array) {
    return float16ToFloat32Array(data);
  }

  // Float16Array is Stage-3 in ECMAScript – guard for environments that expose it.
  if (typeof Float16Array !== 'undefined' && data instanceof Float16Array) {
    const dst = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) dst[i] = data[i];
    return dst;
  }

  return Float32Array.from(data);
}

/**
 * Lightweight Parakeet model wrapper designed for browser usage.
 * Currently supports the *combined* decoder_joint-model ONNX (encoder+decoder+joiner in '
 * transformerjs' style) exported by parakeet TDT.
 *
 * NOTE: This is an *early* scaffold – the `transcribe` method is TODO.
 */
export class ParakeetModel {
  constructor({ tokenizer, encoderSession, joinerSession, preprocessor, ort, subsampling = 8, windowStride = 0.01, normalizer = (s)=>s }) {
    this.tokenizer = tokenizer;
    this.encoderSession = encoderSession;
    this.joinerSession = joinerSession;
    this.preprocessor = preprocessor;
    this.ort = ort;

    // Default IDs – may later be read from model metadata.
    this.blankId = 1024;

    // Combined model specific constants
    this.predHidden = 640;
    this.predLayers = 2;
    this.maxTokensPerStep = 10;

    // Allocate zero LSTM states for the combined decoder; will be reused.
    const numLayers = this.predLayers;
    const hidden = this.predHidden;
    const size = numLayers * 1 * hidden;
    const z = new Float32Array(size); // zeros
    this._combState1 = new ort.Tensor('float32', z, [numLayers, 1, hidden]);
    this._combState2 = new ort.Tensor('float32', z.slice(), [numLayers, 1, hidden]);

    this._normalizer = normalizer;
    this.subsampling = subsampling;
    this.windowStride = windowStride;
  }

  /**
   * Create ParakeetModel by downloading all required assets.
   * @param {Object} cfg
   * @param {string} cfg.encoderUrl URL to encoder-model.onnx
   * @param {string} cfg.decoderUrl URL to decoder_joint-model.onnx
   * @param {string} cfg.tokenizerUrl URL to vocab.txt or tokens.txt
   * @param {string} cfg.preprocessorUrl URL to nemo80/128.onnx
   * @param {('webgpu'|'wasm')} [cfg.backend='webgpu']
   */
  static async fromUrls(cfg) {
    const {
      encoderUrl,
      decoderUrl,
      tokenizerUrl,
      preprocessorUrl,
      encoderDataUrl,
      decoderDataUrl,
      filenames,
      backend = 'webgpu-hybrid',
      wasmPaths,
      subsampling = 8,
      windowStride = 0.01,
      verbose = false,
      enableProfiling = false,
      enableGraphCapture,
      cpuThreads = undefined,
    } = cfg;

    if (!encoderUrl || !decoderUrl || !tokenizerUrl || !preprocessorUrl) {
      throw new Error('fromUrls requires encoderUrl, decoderUrl, tokenizerUrl and preprocessorUrl');
    }

    // 1. Init ONNX Runtime
    let ortBackend = backend;
    if (backend.startsWith('webgpu')) {
        ortBackend = 'webgpu';
    }
    const ort = await initOrt({ backend: ortBackend, wasmPaths, numThreads: cpuThreads });

    // 2. Configure session options for better performance
    // Graph-capture is beneficial only when every node runs on the same EP and
    // ORT can fully record the graph (currently true only for a “strict”
    // WebGPU session).  We therefore enable it *only* when the caller passes
    // `enableGraphCapture:true` **and** the selected backend is the strict
    // WebGPU preset.  In all other scenarios (hybrid WebGPU or pure WASM)
    // it is forced off to avoid the “External buffer must be provided …”
    // runtime error on recent ORT builds.
    const graphCaptureEnabled = !!enableGraphCapture && backend === 'webgpu-strict';
    const isFullWasm = backend === 'wasm';

    const baseSessionOptions = {
      executionProviders: [],
      graphOptimizationLevel: 'all',
      executionMode: 'parallel',
      enableCpuMemArena: true,
      enableMemPattern: true,
      enableProfiling,
      enableGraphCapture: graphCaptureEnabled,
      logSeverityLevel: verbose ? 0 : 2, // 0=verbose, 2=warning
    };

    // Set execution provider based on backend
    if (backend === 'webgpu-hybrid') {
      // WebGPU with fallback to WASM for encoder; decoder may be forced to WASM-only.
      baseSessionOptions.executionProviders = [
        {
          name: 'webgpu',
          deviceType: 'gpu',
          powerPreference: 'high-performance'
        },
        'wasm'
      ];
    } else if (backend === 'webgpu-strict') {
      baseSessionOptions.executionProviders = [
        {
          name: 'webgpu',
          deviceType: 'gpu',
          powerPreference: 'high-performance'
        }
      ];
    } else if (backend === 'wasm') {
      baseSessionOptions.executionProviders = ['wasm'];
    }

    console.log(`[Parakeet.js] Creating ONNX sessions with execution mode '${backend}'. Providers:`, baseSessionOptions.executionProviders);
    if (verbose) {
        console.log('[Parakeet.js] Verbose logging enabled for ONNX Runtime.');
    }

    // Create separate options for sessions that might have external data
    const encoderSessionOptions = { ...baseSessionOptions };
    if (encoderDataUrl && filenames?.encoder) {
        encoderSessionOptions.externalData = [{
            data: encoderDataUrl,
            path: filenames.encoder + '.data',
        }];
    }

    const decoderSessionOptions = { ...baseSessionOptions };
    if (decoderDataUrl && filenames?.decoder) {
        decoderSessionOptions.externalData = [{
            data: decoderDataUrl,
            path: filenames.decoder + '.data',
        }];
    }

    // In hybrid mode, the decoder is always run on WASM to avoid per-step
    // stalls. In pure WASM mode, both EPs are WASM anyway.
    if (backend.startsWith('webgpu')) {
      // Force decoder to run on WASM
      decoderSessionOptions.executionProviders = ['wasm'];
    }

    // 3. Load tokenizer & preprocessor in parallel with model sessions
    // helper to create session with graceful fallback if graph capture is unsupported
    async function createSession(url, opts) {
      try {
        return await ort.InferenceSession.create(url, opts);
      } catch (e) {
        const msg = (e.message || '') + '';
        if (opts.enableGraphCapture && msg.includes('graph capture')) {
          console.warn('[Parakeet] Graph-capture unsupported for this model/backend; retrying without it');
          const retryOpts = { ...opts, enableGraphCapture: false };
          return await ort.InferenceSession.create(url, retryOpts);
        }
        throw e;
      }
    }

    const tokenizerPromise = ParakeetTokenizer.fromUrl(tokenizerUrl);
    const preprocPromise = Promise.resolve(new OnnxPreprocessor(preprocessorUrl, { backend, wasmPaths, enableProfiling, enableGraphCapture: isFullWasm ? false : graphCaptureEnabled, numThreads: cpuThreads }));

    let encoderSession, joinerSession;
    if (backend === 'webgpu-hybrid') {
      // avoid parallel create to prevent double initWasm race
      encoderSession = await createSession(encoderUrl, encoderSessionOptions);
      joinerSession = await createSession(decoderUrl, decoderSessionOptions);
    } else {
      [encoderSession, joinerSession] = await Promise.all([
        createSession(encoderUrl, encoderSessionOptions),
        createSession(decoderUrl, decoderSessionOptions),
      ]);
    }

    const [tokenizer, preprocessor] = await Promise.all([tokenizerPromise, preprocPromise]);

    return new ParakeetModel({ tokenizer, encoderSession, joinerSession, preprocessor, ort, subsampling, windowStride });
  }

  async _runCombinedStep(encTensor, token, currentState = null) {
    const singleToken = typeof token === 'number' ? token : this.blankId;

    const targetTensor = new this.ort.Tensor('int32', new Int32Array([singleToken]), [1, 1]);
    const lenTensor = new this.ort.Tensor('int32', new Int32Array([1]), [1]);

    const state1 = currentState?.state1 || this._combState1;
    const state2 = currentState?.state2 || this._combState2;

    const feeds = {
      encoder_outputs: encTensor,
      targets: targetTensor,
      target_length: lenTensor,
      input_states_1: state1,
      input_states_2: state2,
    };

    const out = await this.joinerSession.run(feeds);
    const logits = out['outputs'] ?? Object.values(out)[0];
    if (!logits) {
      throw new Error('Decoder session did not return outputs tensor');
    }

    const vocab = this.tokenizer.id2token.length;
    const totalDim = logits.dims[3];
    const data = tensorDataToFloat32(logits);
    if (!data) {
      throw new Error('Decoder outputs missing data buffer');
    }

    const tokenLogits = data.slice(0, vocab);
    const durLogits = data.slice(vocab, totalDim);

    let step = 0;
    if (durLogits.length) {
      let maxVal = -Infinity;
      for (let i = 0; i < durLogits.length; ++i) if (durLogits[i] > maxVal) { maxVal = durLogits[i]; step = i; }
    }

    const newState = {
      state1: out['output_states_1'] || state1,
      state2: out['output_states_2'] || state2,
    };

    return { tokenLogits, step, newState };
  }

  async computeFeatures(audio, sampleRate = 16000) {
    const { features, length } = await this.preprocessor.process(audio);
    const T = length; // number of frames returned by preprocessor
    const melBins = features.length / T;
    return { features, T, melBins };
  }

  /**
   * Transcribe 16-kHz mono PCM. Returns full rich output (timestamps/confidences opt-in).
   */
  async transcribe(audio, sampleRate = 16000, opts = {}) {
    const {
      returnTimestamps = false,
      returnConfidences = false,
      temperature = 1.2,
      debug = false,
      skipCMVN = false,
      frameStride = 1,
      encoderCache = null,
    } = opts;

    const perfEnabled = true; // always collect and log timings
    let t0, tPreproc = 0, tEncode = 0, tDecode = 0, tToken = 0;
    if (perfEnabled) t0 = performance.now();

    // 1. Feature extraction (ONNX pre-processor)
    let features, T, melBins;
    if (perfEnabled) {
      const s = performance.now();
      ({ features, T, melBins } = await this.computeFeatures(audio, sampleRate));
      tPreproc = performance.now() - s;
    } else {
      ({ features, T, melBins } = await this.computeFeatures(audio, sampleRate));
    }

    // 2. Encode utterance (optionally using a cached encoder output)
    let cachedTensor = null;
    let cachedFrames = 0;
    if (encoderCache) {
      const dims = encoderCache.dims;
      if (Array.isArray(dims) && dims.length === 3 && dims[0] === 1) {
        cachedTensor = encoderCache;
        cachedFrames = Math.max(0, dims[2] | 0);
      } else if (debug) {
        console.warn('[Parakeet] Ignoring encoderCache with invalid shape', dims);
      }
    }

    cachedFrames = Math.min(cachedFrames, T);
    const framesToEncode = Math.max(0, T - cachedFrames);

    const sliceFeatureFrames = (startFrame, frameCount) => {
      if (frameCount <= 0) return new Float32Array(0);
      if (startFrame === 0 && frameCount === T) return features;
      const sliced = new Float32Array(melBins * frameCount);
      for (let m = 0; m < melBins; m++) {
        const srcOffset = m * T + startFrame;
        const dstOffset = m * frameCount;
        sliced.set(features.subarray(srcOffset, srcOffset + frameCount), dstOffset);
      }
      return sliced;
    };

    let encodeDuration = 0;
    const runEncoder = async (featureBuffer, frameCount) => {
      if (frameCount <= 0) return null;
      const inputTensor = new this.ort.Tensor('float32', featureBuffer, [1, melBins, frameCount]);
      const lenTensor = new this.ort.Tensor('int64', BigInt64Array.from([BigInt(frameCount)]), [1]);
      let encOut;
      if (perfEnabled) {
        const s = performance.now();
        encOut = await this.encoderSession.run({ audio_signal: inputTensor, length: lenTensor });
        encodeDuration += performance.now() - s;
      } else {
        encOut = await this.encoderSession.run({ audio_signal: inputTensor, length: lenTensor });
      }
      const tensor = encOut['outputs'] ?? Object.values(encOut)[0];
      if (!tensor) {
        throw new Error('Encoder session did not return outputs tensor');
      }
      return tensor;
    };

    const truncateEncoderCache = (cacheTensor, targetFrames) => {
      const dims = cacheTensor?.dims;
      if (!Array.isArray(dims) || dims.length !== 3 || dims[0] !== 1) return null;
      const [, Dcache, cacheTotalFrames] = dims;
      if (cacheTotalFrames === targetFrames) return cacheTensor;
      if (cacheTotalFrames < targetFrames) return null;
      const cacheData = tensorDataToFloat32(cacheTensor);
      if (!cacheData) return null;
      const truncated = new Float32Array(Dcache * targetFrames);
      for (let d = 0; d < Dcache; d++) {
        const srcOffset = d * cacheTotalFrames;
        const dstOffset = d * targetFrames;
        truncated.set(cacheData.subarray(srcOffset, srcOffset + targetFrames), dstOffset);
      }
      return new this.ort.Tensor('float32', truncated, [1, Dcache, targetFrames]);
    };

    const concatEncoderOutputs = (cacheTensor, newTensor, keepFromCache, totalFrames) => {
      const cacheDims = cacheTensor?.dims;
      const newDims = newTensor?.dims;
      if (!Array.isArray(cacheDims) || !Array.isArray(newDims)) return null;
      if (cacheDims.length !== 3 || newDims.length !== 3) return null;
      if (cacheDims[0] !== 1 || newDims[0] !== 1) return null;
      const [, Dcache, cacheTotalFrames] = cacheDims;
      const [, Dnew, newFrames] = newDims;
      if (Dcache !== Dnew) return null;
      if (keepFromCache > cacheTotalFrames) return null;
      if (newFrames !== totalFrames - keepFromCache) return null;
      const cacheData = tensorDataToFloat32(cacheTensor);
      const newData = tensorDataToFloat32(newTensor);
      if (!cacheData || !newData) return null;
      const combined = new Float32Array(Dcache * totalFrames);
      for (let d = 0; d < Dcache; d++) {
        const dstOffset = d * totalFrames;
        const cacheOffset = d * cacheTotalFrames;
        const newOffset = d * newFrames;
        combined.set(cacheData.subarray(cacheOffset, cacheOffset + keepFromCache), dstOffset);
        combined.set(newData.subarray(newOffset, newOffset + newFrames), dstOffset + keepFromCache);
      }
      return new this.ort.Tensor('float32', combined, [1, Dcache, totalFrames]);
    };

    let encTensor = null;
    if (T === 0) {
      if (cachedTensor) {
        encTensor = truncateEncoderCache(cachedTensor, 0) || cachedTensor;
      }
      if (!encTensor) {
        encTensor = new this.ort.Tensor('float32', new Float32Array(0), [1, 0, 0]);
      }
    } else if (cachedTensor && cachedFrames > 0) {
      if (framesToEncode > 0) {
        const tailFeatures = sliceFeatureFrames(cachedFrames, framesToEncode);
        const newTensor = await runEncoder(tailFeatures, framesToEncode);
        if (newTensor) {
          encTensor = concatEncoderOutputs(cachedTensor, newTensor, cachedFrames, T);
        }
        if (!encTensor) {
          if (debug) {
            console.warn('[Parakeet] Failed to merge encoder cache; re-encoding entire segment');
          }
          encTensor = await runEncoder(features, T);
        }
      } else {
        encTensor = truncateEncoderCache(cachedTensor, T);
        if (!encTensor) {
          if (debug) {
            console.warn('[Parakeet] Unable to reuse encoder cache; re-encoding entire segment');
          }
          encTensor = await runEncoder(features, T);
        }
      }
    } else {
      encTensor = await runEncoder(features, T);
    }

    if (perfEnabled) tEncode = encodeDuration;

    if (!encTensor) {
      throw new Error('Encoder session did not produce outputs tensor');
    }

    const encData = tensorDataToFloat32(encTensor);
    if (!encData) {
      throw new Error('Encoder outputs missing data buffer');
    }
    const [, D, Tenc] = encTensor.dims;
    const transposed = new Float32Array(Tenc * D);
    for (let d = 0; d < D; d++) {
      for (let t = 0; t < Tenc; t++) {
        transposed[t * D + d] = encData[d * Tenc + t];
      }
    }

    // --- Decode frame-by-frame ----------------------------------------
    const ids = [];
    const tokenTimes = [];
    const tokenConfs = [];
    const frameConfs = [];
    let overallLogProb = 0;

    let decoderState = null;
    let emittedTokens = 0;

    const decStartTime = perfEnabled ? performance.now() : 0;

    for (let t = 0; t < Tenc; ) {
      const frameBuf = transposed.subarray(t * D, (t + 1) * D);
      const frameTensor = new this.ort.Tensor('float32', frameBuf, [1, D, 1]);

      const prevTok = ids.length ? ids[ids.length - 1] : this.blankId;
      const { tokenLogits, step, newState } = await this._runCombinedStep(frameTensor, prevTok, decoderState);
      decoderState = newState;

      // Temperature scaling & argmax
      let maxVal = -Infinity, maxId = 0;
      for (let i = 0; i < tokenLogits.length; i++) {
        const v = tokenLogits[i] / temperature;
        if (v > maxVal) { maxVal = v; maxId = i; }
      }
      let sumExp = 0;
      for (let i = 0; i < tokenLogits.length; i++) {
        sumExp += Math.exp((tokenLogits[i] / temperature) - maxVal);
      }
      const confVal = 1 / sumExp;
      frameConfs.push(confVal);
      overallLogProb += Math.log(confVal);

      if (maxId !== this.blankId) {
        ids.push(maxId);
        if (returnTimestamps) {
          const TIME_STRIDE = this.subsampling * this.windowStride;
          const durFrames = step > 0 ? step : 1;
          const start = t * TIME_STRIDE;
          const end = (t + durFrames) * TIME_STRIDE;
          tokenTimes.push([start, end]);
        }
        if (returnConfidences) tokenConfs.push(confVal);
        emittedTokens += 1;
      }

      const shouldAdvance = maxId === this.blankId || emittedTokens >= this.maxTokensPerStep;
      t += step > 0 ? step : (shouldAdvance ? frameStride : 0);
      if (!shouldAdvance && step === 0) t += 1; // safeguard
      if (maxId === this.blankId) emittedTokens = 0;
    }

    if (perfEnabled) {
      tDecode = performance.now() - decStartTime;
    }

    let tokenStart;
    if (perfEnabled) tokenStart = performance.now();
    const text = this._normalizer(this.tokenizer.decode(ids));
    if (perfEnabled) tToken = performance.now() - tokenStart;

    // Early exit if no extras requested
    if (!returnTimestamps && !returnConfidences) {
      if (perfEnabled) {
        const total = performance.now() - t0;
        const audioDur = audio.length / sampleRate;
        const rtf = audioDur / (total / 1000);
        console.log(`[Perf] RTF: ${rtf.toFixed(2)}x (audio ${audioDur.toFixed(2)} s, time ${(total/1000).toFixed(2)} s)`);
        console.table({Preprocess:`${tPreproc.toFixed(1)} ms`, Encode:`${tEncode.toFixed(1)} ms`, Decode:`${tDecode.toFixed(1)} ms`, Tokenize:`${tToken.toFixed(1)} ms`, Total:`${total.toFixed(1)} ms`});
      }
      const metrics = perfEnabled ? {
        preprocess_ms: +tPreproc.toFixed(1),
        encode_ms: +tEncode.toFixed(1),
        decode_ms: +tDecode.toFixed(1),
        tokenize_ms: +tToken.toFixed(1),
        total_ms: +( (performance.now() - t0).toFixed(1) ),
        rtf: +((audio.length / sampleRate) / ((performance.now() - t0) / 1000)).toFixed(2)
      } : null;
      return { utterance_text: text, words: [], metrics, encoder_output: encTensor, is_final: true };
    }

    // --- Build words & detailed token arrays ---------------------------
    const words = [];
    const tokensDetailed = [];
    let currentWord = '', wordStart = 0, wordEnd = 0;
    let wordConfs = [];

    ids.forEach((tokId, i) => {
      const raw = this.tokenizer.id2token[tokId];
      if (raw === this.tokenizer.blankToken) return;

      const isWordStart = raw.startsWith('▁');
      const cleanTok = isWordStart ? raw.slice(1) : raw;
      const ts = tokenTimes[i] || [null, null];
      const conf = tokenConfs[i];

      // tokensDetailed entry
      const tokEntry = { token: [cleanTok] };
      if (returnTimestamps) { tokEntry.start_time = +ts[0].toFixed(3); tokEntry.end_time = +ts[1].toFixed(3); }
      if (returnConfidences) tokEntry.confidence = +conf.toFixed(4);
      tokensDetailed.push(tokEntry);

      // accumulate into words
      if (isWordStart) {
        if (currentWord) {
          const avg = wordConfs.length ? wordConfs.reduce((a,b)=>a+b,0)/wordConfs.length : 0;
          words.push({ text: currentWord, start_time: +wordStart.toFixed(3), end_time: +wordEnd.toFixed(3), confidence: +avg.toFixed(4) });
        }
        currentWord = cleanTok;
        if (returnTimestamps) { wordStart = ts[0]; wordEnd = ts[1]; }
        wordConfs = returnConfidences ? [conf] : [];
      } else {
        currentWord += cleanTok;
        if (returnTimestamps) wordEnd = ts[1];
        if (returnConfidences) wordConfs.push(conf);
      }
    });

    if (currentWord) {
      const avg = wordConfs.length ? wordConfs.reduce((a,b)=>a+b,0)/wordConfs.length : 0;
      words.push({ text: currentWord, start_time: +wordStart.toFixed(3), end_time: +wordEnd.toFixed(3), confidence: +avg.toFixed(4) });
    }

    const avgWordConf = words.length && returnConfidences ? words.reduce((a,b)=>a+b.confidence,0)/words.length : null;
    const avgTokenConf = tokensDetailed.length && returnConfidences ? tokensDetailed.reduce((a,b)=>a+(b.confidence||0),0)/tokensDetailed.length : null;

    if (perfEnabled) {
      const total = performance.now() - t0;
      const audioDur = audio.length / sampleRate;
      const rtf = audioDur / (total / 1000);
      console.log(`[Perf] RTF: ${rtf.toFixed(2)}x (audio ${audioDur.toFixed(2)} s, time ${(total/1000).toFixed(2)} s)`);
      console.table({Preprocess:`${tPreproc.toFixed(1)} ms`, Encode:`${tEncode.toFixed(1)} ms`, Decode:`${tDecode.toFixed(1)} ms`, Tokenize:`${tToken.toFixed(1)} ms`, Total:`${total.toFixed(1)} ms`});
    }

    return {
      utterance_text: text,
      words,
      tokens: tokensDetailed,
      confidence_scores: returnConfidences ? {
        token: tokenConfs.map(c=>+c.toFixed(4)),
        token_avg: +avgTokenConf?.toFixed(4),
        word: words.map(w=>w.confidence),
        word_avg: +avgWordConf?.toFixed(4),
        frame: frameConfs.map(f=>+f.toFixed(4)),
        frame_avg: frameConfs.length ? +(frameConfs.reduce((a,b)=>a+b,0)/frameConfs.length).toFixed(4) : null,
        overall_log_prob: +overallLogProb.toFixed(6)
      } : { overall_log_prob: null, frame: null, frame_avg: null },
      metrics: perfEnabled ? {
        preprocess_ms: +tPreproc.toFixed(1),
        encode_ms: +tEncode.toFixed(1),
        decode_ms: +tDecode.toFixed(1),
        tokenize_ms: +tToken.toFixed(1),
        total_ms: +( (performance.now() - t0).toFixed(1) ),
        rtf: +((audio.length / sampleRate) / ((performance.now() - t0) / 1000)).toFixed(2)
      } : null,
      encoder_output: encTensor,
      is_final: true,
    };
  }

  /**
   * Stop ORT profiling (if enabled) for all sessions and print a quick summary
   * of time spent on GPU (WebGPU) vs CPU (WASM) kernels. Returns the parsed
   * summary object for further inspection.
   */
  endProfiling() {
    try { this.encoderSession?.endProfiling(); } catch(e) { /* ignore */ }
    try { this.joinerSession?.endProfiling(); } catch(e) { /* ignore */ }

    const FS = this.ort?.env?.wasm?.FS;
    if (!FS) {
      console.warn('[Parakeet] Profiling FS not accessible');
      return null;
    }

    const files = FS.readdir('/tmp').filter(f => f.startsWith('profile_') && f.endsWith('.json'));
    if (!files.length) {
      console.warn('[Parakeet] No profiling files found. Was profiling enabled?');
      return null;
    }

    const summary = {};
    for (const file of files) {
      try {
        const txt = FS.readFile('/tmp/' + file, { encoding: 'utf8' });
        const events = JSON.parse(txt);
        let gpu = 0, cpu = 0;
        for (const ev of events) {
          if (ev.cat === 'Node') {
            const prov = ev.args?.provider;
            if (prov === 'webgpu') gpu += ev.dur;
            else if (prov) cpu += ev.dur;
          }
        }
        summary[file] = { gpu_us: gpu, cpu_us: cpu, total_us: gpu + cpu };
      } catch (err) {
        console.warn('[Parakeet] Failed to parse profile file', file, err);
      }
    }
    console.table(summary);
    return summary;
  }
} 
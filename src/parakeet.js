import { initOrt } from './backend.js';
import { ParakeetTokenizer } from './tokenizer.js';
import { OnnxPreprocessor } from './preprocessor.js';

/**
 * Lightweight Parakeet model wrapper designed for browser usage.
 * Currently supports the *combined* decoder_joint-model ONNX (encoder+decoder+joiner in '
 * transformerjs' style) exported by parakeet TDT.
 *
 * NOTE: This is an *early* scaffold – the `transcribe` method is TODO.
 */
export class ParakeetModel {
  constructor({ tokenizer, encoderSession, joinerSession, preprocessor, ort, subsampling = 8, windowStride = 0.01, normalizer }) {
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

    // Decoder state bookkeeping for the combined model.
    const numLayers = this.predLayers;
    const hidden = this.predHidden;
    this._decoderStateShape = [numLayers, 1, hidden];
    this._decoderStateSize = numLayers * hidden;

    this._normalizer = typeof normalizer === 'function' ? normalizer : (s) => s;
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

  _createZeroDecoderState() {
    const state1 = new this.ort.Tensor('float32', new Float32Array(this._decoderStateSize), this._decoderStateShape);
    const state2 = new this.ort.Tensor('float32', new Float32Array(this._decoderStateSize), this._decoderStateShape);
    return { state1, state2 };
  }

  async _runCombinedStep(encTensor, token, currentState) {
    const singleToken = typeof token === 'number' ? token : this.blankId;

    const targetTensor = new this.ort.Tensor('int32', new Int32Array([singleToken]), [1, 1]);
    const lenTensor = new this.ort.Tensor('int32', new Int32Array([1]), [1]);

    const state1 = currentState.state1;
    const state2 = currentState.state2;

    const feeds = {
      encoder_outputs: encTensor,
      targets: targetTensor,
      target_length: lenTensor,
      input_states_1: state1,
      input_states_2: state2,
    };

    const out = await this.joinerSession.run(feeds);
    const logits = out['outputs'];

    const vocab = this.tokenizer.id2token.length;
    const totalDim = logits.dims[3];
    const data = logits.data;

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
      temperature = 1.0,
      debug = false,
      skipCMVN = false,
      frameStride = 1,
      ngramBias: optNgramBias,
      ngramAlpha: optNgramAlpha,
      blankPenalty: optBlankPenalty,
    } = opts;

    const ngramBias = optNgramBias ?? opts.ngram_bias ?? null;
    const ngramAlpha = (typeof optNgramAlpha === 'number' ? optNgramAlpha : undefined) ??
      (typeof opts.ngram_alpha === 'number' ? opts.ngram_alpha : 0.1);
    const blankPenalty = (typeof optBlankPenalty === 'number' ? optBlankPenalty : undefined) ??
      (typeof opts.blank_penalty === 'number' ? opts.blank_penalty : 0);

    const lookupNgramBias = (biasTree, history, candidateId) => {
      if (!biasTree) return null;
      const histLength = history.length;

      for (let n = histLength; n >= 0; n--) {
        let node = biasTree;
        let valid = true;

        if (n > 0) {
          const start = histLength - n;
          for (let i = start; i < histLength; i++) {
            node = node?.[history[i]];
            if (!node || typeof node !== 'object') {
              valid = false;
              break;
            }
          }
          if (!valid) continue;
        }

        const candidateNode = node?.[candidateId];
        if (candidateNode && typeof candidateNode === 'object' && typeof candidateNode.log_prob === 'number') {
          return candidateNode.log_prob;
        }
      }

      return null;
    };

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

    // 2. Encode entire utterance
    const input = new this.ort.Tensor('float32', features, [1, melBins, T]);
    const lenTensor = new this.ort.Tensor('int64', BigInt64Array.from([BigInt(T)]), [1]);
    let enc;
    if (perfEnabled) {
      const s = performance.now();
      const encOut = await this.encoderSession.run({ audio_signal: input, length: lenTensor });
      tEncode = performance.now() - s;
      enc = encOut['outputs'] ?? Object.values(encOut)[0];
    } else {
      const encOut = await this.encoderSession.run({ audio_signal: input, length: lenTensor });
      enc = encOut['outputs'] ?? Object.values(encOut)[0];
    }

    // Transpose encoder output [B, D, T] ➔ [T, D] for B=1
    const [ , D, Tenc ] = enc.dims;
    const transposed = new Float32Array(Tenc * D);
    for (let d = 0; d < D; d++) {
      for (let t = 0; t < Tenc; t++) {
        transposed[t * D + d] = enc.data[d * Tenc + t];
      }
    }

    // --- Decode frame-by-frame ----------------------------------------
    const ids = [];
    const tokenTimes = [];
    const tokenConfs = [];
    const frameConfs = [];
    let overallLogProb = 0;

    let decoderState = this._createZeroDecoderState();
    let emittedTokens = 0;

    const decStartTime = perfEnabled ? performance.now() : 0;

    for (let t = 0; t < Tenc; ) {
      const frameBuf = transposed.subarray(t * D, (t + 1) * D);
      const encTensor = new this.ort.Tensor('float32', frameBuf, [1, D, 1]);

      const prevTok = ids.length ? ids[ids.length - 1] : this.blankId;
      const { tokenLogits, step, newState } = await this._runCombinedStep(encTensor, prevTok, decoderState);

      // Temperature scaling & argmax
      if (blankPenalty !== 0 && this.blankId < tokenLogits.length) {
        tokenLogits[this.blankId] -= blankPenalty;
      }
      let maxVal = -Infinity, maxId = 0;
      for (let i = 0; i < tokenLogits.length; i++) {
        if (ngramBias && i !== this.blankId) {
          const biasLogProb = lookupNgramBias(ngramBias, ids, i);
          if (typeof biasLogProb === 'number') {
            tokenLogits[i] += ngramAlpha * biasLogProb;
          }
        }
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

      const emittedNonBlank = maxId !== this.blankId;
      if (emittedNonBlank) {
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
        decoderState = newState;
      }

      if (step > 0) {
        t += step;
        emittedTokens = 0;
      } else if (maxId === this.blankId || emittedTokens >= this.maxTokensPerStep) {
        t += frameStride;
        emittedTokens = 0;
      }
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
      return { utterance_text: text, words: [], metrics, is_final: true };
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
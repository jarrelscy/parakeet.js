const LOG_FLOOR_DEFAULT = 1e-10;
const LOG_FLOOR_MEDASR = 1e-5;

// HTK mel scale (default for Parakeet models)
function hzToMelHtk(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHzHtk(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

// Kaldi mel scale (used by MedASR/HuggingFace LasrFeatureExtractor)
function hzToMelKaldi(hz) {
  return 1127.0 * Math.log(1.0 + hz / 700.0);
}

function hannWindow(length) {
  const window = new Float32Array(length);
  const scale = 2 * Math.PI / (length - 1);
  for (let i = 0; i < length; i++) {
    window[i] = 0.5 - 0.5 * Math.cos(scale * i);
  }
  return window;
}

// Default mel filterbank (HTK scale, used by Parakeet models)
function createMelFilterbank(nMels, nFft, sampleRate, fMin = 0, fMax = sampleRate / 2) {
  const fftBins = nFft / 2 + 1;
  const melMin = hzToMelHtk(fMin);
  const melMax = hzToMelHtk(fMax);
  const melPoints = [];
  const step = (melMax - melMin) / (nMels + 1);
  for (let i = 0; i < nMels + 2; i++) {
    melPoints.push(melMin + step * i);
  }

  const hzPoints = melPoints.map((mel) => melToHzHtk(mel));
  const binPoints = hzPoints.map((hz) => Math.floor((nFft + 1) * hz / sampleRate));

  const filters = new Float32Array(nMels * fftBins);
  for (let m = 0; m < nMels; m++) {
    const left = binPoints[m];
    let center = binPoints[m + 1];
    let right = binPoints[m + 2];
    if (center === left) center += 1;
    if (right === center) right += 1;

    for (let k = left; k < center; k++) {
      if (k >= 0 && k < fftBins) {
        filters[m * fftBins + k] = (k - left) / Math.max(center - left, 1);
      }
    }
    for (let k = center; k < right; k++) {
      if (k >= 0 && k < fftBins) {
        filters[m * fftBins + k] = (right - k) / Math.max(right - center, 1);
      }
    }
  }
  return { filters, fftBins };
}

/**
 * Create mel filterbank matching HuggingFace's linear_to_mel_weight_matrix.
 * This uses Kaldi mel scale and excludes DC bin.
 * Used by MedASR (google/medasr) model.
 */
function createMelFilterbankMedASR(nMels, nFft, sampleRate, lowerEdgeHz = 125.0, upperEdgeHz = 7500.0) {
  const numSpectrogramBins = nFft / 2 + 1;
  const bandsToZero = 1;  // Excludes DC bin
  const nyquist = sampleRate / 2.0;

  // Linear frequencies (excluding DC)
  const linearFreqs = [];
  for (let i = bandsToZero; i < numSpectrogramBins; i++) {
    linearFreqs.push((i / (numSpectrogramBins - 1)) * nyquist);
  }

  // Convert linear frequencies to mel (Kaldi scale)
  const spectrogramBinsMel = linearFreqs.map(hz => hzToMelKaldi(hz));

  // Mel band edges
  const lowerMel = hzToMelKaldi(lowerEdgeHz);
  const upperMel = hzToMelKaldi(upperEdgeHz);
  const edges = [];
  for (let i = 0; i < nMels + 2; i++) {
    edges.push(lowerMel + (upperMel - lowerMel) * i / (nMels + 1));
  }

  // Create filterbank with shape [numSpectrogramBins, nMels] for matmul compatibility
  const filters = new Float64Array(numSpectrogramBins * nMels);

  for (let m = 0; m < nMels; m++) {
    const lowerEdgeMel = edges[m];
    const centerMel = edges[m + 1];
    const upperEdgeMel = edges[m + 2];

    for (let i = 0; i < spectrogramBinsMel.length; i++) {
      const binMel = spectrogramBinsMel[i];
      const lowerSlope = (binMel - lowerEdgeMel) / (centerMel - lowerEdgeMel);
      const upperSlope = (upperEdgeMel - binMel) / (upperEdgeMel - centerMel);
      const weight = Math.max(0.0, Math.min(lowerSlope, upperSlope));
      // Account for bands_to_zero offset - store in [bin, mel] order
      filters[(i + bandsToZero) * nMels + m] = weight;
    }
  }

  return { filters, fftBins: numSpectrogramBins };
}

function fftRadix2(re, im) {
  const n = re.length;
  for (let i = 0, j = 0; i < n; i++) {
    if (j > i) {
      const tmpRe = re[j];
      const tmpIm = im[j];
      re[j] = re[i];
      im[j] = im[i];
      re[i] = tmpRe;
      im[i] = tmpIm;
    }
    let m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const tableStep = n / size;
    for (let i = 0; i < n; i += size) {
      for (let j = 0; j < half; j++) {
        const k = j * tableStep;
        const angle = -2 * Math.PI * k / n;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const tre = re[i + j + half] * cos - im[i + j + half] * sin;
        const tim = re[i + j + half] * sin + im[i + j + half] * cos;
        re[i + j + half] = re[i + j] - tre;
        im[i + j + half] = im[i + j] - tim;
        re[i + j] += tre;
        im[i + j] += tim;
      }
    }
  }
}

export class LogMelPreprocessor {
  /**
   * @param {Object} config - Configuration options
   * @param {number} [config.sampleRate=16000] - Audio sample rate
   * @param {number} [config.nFft=512] - FFT size
   * @param {number} [config.winLength=400] - Window length in samples
   * @param {number} [config.hopLength=160] - Hop length in samples
   * @param {number} [config.nMels=128] - Number of mel bins
   * @param {number} [config.paddingValue=0.0] - Padding value
   * @param {string} [config.paddingSide='right'] - Padding side
   * @param {boolean} [config.medasr=false] - Use MedASR-compatible preprocessing
   */
  constructor(config) {
    this.sampleRate = config.sampleRate ?? 16000;
    this.nFft = config.nFft ?? 512;
    this.winLength = config.winLength ?? 400;
    this.hopLength = config.hopLength ?? 160;
    this.nMels = config.nMels ?? config.featureSize ?? 128;
    this.paddingValue = config.paddingValue ?? 0.0;
    this.paddingSide = config.paddingSide ?? 'right';
    this.medasr = config.medasr ?? false;

    this.window = hannWindow(this.winLength);

    if (this.medasr) {
      // MedASR uses Kaldi mel scale with 125-7500 Hz range
      const { filters, fftBins } = createMelFilterbankMedASR(
        this.nMels,
        this.nFft,
        this.sampleRate,
        125.0,
        7500.0
      );
      this.melFilters = filters;
      this.fftBins = fftBins;
      this.logFloor = LOG_FLOOR_MEDASR;
    } else {
      // Default: HTK mel scale with 0-Nyquist range
      const { filters, fftBins } = createMelFilterbank(this.nMels, this.nFft, this.sampleRate);
      this.melFilters = filters;
      this.fftBins = fftBins;
      this.logFloor = LOG_FLOOR_DEFAULT;
    }
  }

  /**
   * Load preprocessor config from a URL.
   * @param {string} configUrl - URL to processor_config.json
   * @param {Object} [options] - Additional options
   * @param {boolean} [options.medasr=false] - Use MedASR-compatible preprocessing
   */
  static async fromConfigUrl(configUrl, options = {}) {
    const resp = await fetch(configUrl);
    if (!resp.ok) {
      throw new Error(`Failed to fetch preprocessor config: ${resp.status} ${resp.statusText}`);
    }
    const json = await resp.json();
    return new LogMelPreprocessor({
      sampleRate: json.sampling_rate,
      nFft: json.n_fft,
      winLength: json.win_length,
      hopLength: json.hop_length,
      nMels: json.feature_size,
      paddingValue: json.padding_value,
      paddingSide: json.padding_side,
      medasr: options.medasr ?? false,
    });
  }

  async process(audio) {
    const buffer = new Float32Array(audio);
    const totalSamples = buffer.length;
    const hop = this.hopLength;
    const win = this.winLength;

    let frames;
    let padded;

    if (this.medasr) {
      // MedASR: Use unfold logic - no partial frame padding
      // frames = (length - win_length) / hop_length + 1
      frames = Math.floor((totalSamples - win) / hop) + 1;
      if (frames <= 0) {
        throw new Error(`Audio too short: ${totalSamples} samples, need at least ${win}`);
      }
      padded = buffer;
    } else {
      // Default: Include partial frames with padding
      frames = 1;
      if (totalSamples > win) {
        frames = Math.floor((totalSamples - win) / hop) + 1;
        const remainder = (totalSamples - win) % hop;
        if (remainder !== 0) frames += 1;
      }

      const neededSamples = (frames - 1) * hop + win;
      padded = new Float32Array(Math.max(neededSamples, win));
      padded.set(buffer);
      if (this.paddingSide !== 'right' && totalSamples < neededSamples) {
        padded.fill(this.paddingValue);
        padded.set(buffer, neededSamples - totalSamples);
      } else if (totalSamples < neededSamples) {
        padded.fill(this.paddingValue, totalSamples);
      }
    }

    const features = new Float32Array(frames * this.nMels);

    if (this.medasr) {
      // MedASR: Use Float64 for higher precision matching HuggingFace
      const re = new Float64Array(this.nFft);
      const im = new Float64Array(this.nFft);
      const window64 = new Float64Array(this.window);

      for (let frame = 0; frame < frames; frame++) {
        const offset = frame * hop;
        re.fill(0);
        im.fill(0);

        // Apply window
        for (let i = 0; i < win; i++) {
          re[i] = padded[offset + i] * window64[i];
        }

        fftRadix2(re, im);

        // Compute power spectrum and apply mel filterbank
        // MedASR filterbank is [fftBins, nMels], so we do power @ filters
        for (let m = 0; m < this.nMels; m++) {
          let sum = 0;
          for (let k = 0; k < this.fftBins; k++) {
            const power = re[k] * re[k] + im[k] * im[k];
            sum += power * this.melFilters[k * this.nMels + m];
          }
          features[frame * this.nMels + m] = Math.log(Math.max(sum, this.logFloor));
        }
      }
    } else {
      // Default processing
      const re = new Float32Array(this.nFft);
      const im = new Float32Array(this.nFft);

      for (let frame = 0; frame < frames; frame++) {
        const offset = frame * hop;
        re.fill(0);
        im.fill(0);
        for (let i = 0; i < win; i++) {
          re[i] = padded[offset + i] * this.window[i];
        }

        fftRadix2(re, im);

        for (let m = 0; m < this.nMels; m++) {
          let sum = 0;
          const base = m * this.fftBins;
          for (let k = 0; k < this.fftBins; k++) {
            const real = re[k];
            const imag = im[k];
            const power = real * real + imag * imag;
            sum += power * this.melFilters[base + k];
          }
          features[frame * this.nMels + m] = Math.log(Math.max(sum, this.logFloor));
        }
      }
    }

    return {
      features,
      length: frames,
      featureDim: this.nMels,
      layout: 'BTC',
    };
  }
}

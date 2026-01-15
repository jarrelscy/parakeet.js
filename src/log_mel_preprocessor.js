const LOG_FLOOR = 1e-10;

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function hannWindow(length) {
  const window = new Float32Array(length);
  const scale = 2 * Math.PI / (length - 1);
  for (let i = 0; i < length; i++) {
    window[i] = 0.5 - 0.5 * Math.cos(scale * i);
  }
  return window;
}

function createMelFilterbank(nMels, nFft, sampleRate, fMin = 0, fMax = sampleRate / 2) {
  const fftBins = nFft / 2 + 1;
  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);
  const melPoints = [];
  const step = (melMax - melMin) / (nMels + 1);
  for (let i = 0; i < nMels + 2; i++) {
    melPoints.push(melMin + step * i);
  }

  const hzPoints = melPoints.map((mel) => melToHz(mel));
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
  constructor(config) {
    this.sampleRate = config.sampleRate ?? 16000;
    this.nFft = config.nFft ?? 512;
    this.winLength = config.winLength ?? 400;
    this.hopLength = config.hopLength ?? 160;
    this.nMels = config.nMels ?? config.featureSize ?? 128;
    this.paddingValue = config.paddingValue ?? 0.0;
    this.paddingSide = config.paddingSide ?? 'right';

    this.window = hannWindow(this.winLength);
    const { filters, fftBins } = createMelFilterbank(this.nMels, this.nFft, this.sampleRate);
    this.melFilters = filters;
    this.fftBins = fftBins;
  }

  static async fromConfigUrl(configUrl) {
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
    });
  }

  async process(audio) {
    const buffer = new Float32Array(audio);
    const totalSamples = buffer.length;
    const hop = this.hopLength;
    const win = this.winLength;

    let frames = 1;
    if (totalSamples > win) {
      frames = Math.floor((totalSamples - win) / hop) + 1;
      const remainder = (totalSamples - win) % hop;
      if (remainder !== 0) frames += 1;
    }

    const neededSamples = (frames - 1) * hop + win;
    const padded = new Float32Array(Math.max(neededSamples, win));
    padded.set(buffer);
    if (this.paddingSide !== 'right' && totalSamples < neededSamples) {
      padded.fill(this.paddingValue);
      padded.set(buffer, neededSamples - totalSamples);
    } else if (totalSamples < neededSamples) {
      padded.fill(this.paddingValue, totalSamples);
    }

    const features = new Float32Array(frames * this.nMels);
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
        features[frame * this.nMels + m] = Math.log(Math.max(sum, LOG_FLOOR));
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

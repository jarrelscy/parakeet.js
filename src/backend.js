// Back-end initialisation helper for ONNX Runtime Web.
// At runtime the caller can specify preferred backend ("webgpu", "wasm").
// The function resolves once ONNX Runtime is ready and returns the `ort` module.

/**
 * Initialise ONNX Runtime Web and pick the execution provider.
 * If WebGPU is requested but not supported, we transparently fall back to WASM.
 * @param {Object} opts
 * @param {('webgpu'|'wasm')} [opts.backend='webgpu'] Desired backend.
 * @param {string} [opts.wasmPaths] Optional path prefix for WASM binaries.
 * @returns {Promise<typeof import('onnxruntime-web').default>}
 */
let selectedBackend = null;

export function getSelectedBackend() {
  return selectedBackend;
}

export async function initOrt({ backend = 'webgpu', wasmPaths, numThreads } = {}) {
  let ortModule;
  try {
    ortModule = await import('onnxruntime-web');
  } catch (error) {
    throw new Error('Failed to load ONNX Runtime Web. Please check your network connection.');
  }

  let ort = ortModule.default || ortModule;
  if (!ort.env && ortModule.ort) {
    ort = ortModule.ort;
  }

  if (!ort || !ort.env) {
    throw new Error('ONNX Runtime Web loaded but env is not available. This might be a bundling issue.');
  }

  if (wasmPaths) {
    ort.env.wasm.wasmPaths = wasmPaths;
  }

  const hasNavigator = typeof navigator !== 'undefined';
  const nav = hasNavigator ? navigator : undefined;
  const threadCount = typeof numThreads === 'number' && numThreads > 0
    ? numThreads
    : nav?.hardwareConcurrency || 4;

  if (backend === 'wasm' || backend === 'webgpu') {
    if (typeof SharedArrayBuffer !== 'undefined') {
      ort.env.wasm.numThreads = threadCount;
      ort.env.wasm.simd = true;
    } else {
      ort.env.wasm.numThreads = Math.max(1, threadCount | 0);
    }

    ort.env.wasm.proxy = false;
  }

  if (backend === 'webgpu') {
    const webgpuSupported = hasNavigator && 'gpu' in navigator;
    if (!webgpuSupported) {
      backend = 'wasm';
    }
  }

  selectedBackend = backend;
  return ort;
}

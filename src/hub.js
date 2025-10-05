/**
 * Simplified HuggingFace Hub utilities for parakeet.js
 * Downloads models from HF and caches them in browser storage.
 */

const DB_NAME = 'parakeet-cache-db';
const STORE_NAME = 'file-store';
let dbPromise = null;

// Cache for repo file listings so we only hit the HF API once per page load
const repoFileCache = new Map();

function getAccessToken() {
  const globalToken = typeof globalThis !== 'undefined' ? globalThis.HF_ACCESS_TOKEN : undefined;
  const envToken = typeof process !== 'undefined' && process?.env ? process.env.HF_ACCESS_TOKEN : undefined;
  return envToken || globalToken || '';
}

async function fetchWithAuth(url, options = {}) {
  const token = getAccessToken();
  const target = url instanceof URL ? url.toString() : url;
  const isHttp = typeof target === 'string' ? /^https?:/i.test(target) : false;
  if (!token || !isHttp) {
    try {
      return await fetch(url, options);
    } catch (err) {
      if (await shouldTryNodeFetch(err)) {
        return nodeFetch(target, options);
      }
      throw err;
    }
  }

  const nextOptions = { ...options };

  if (typeof Headers !== 'undefined') {
    const headers = new Headers(options.headers || undefined);
    if (!headers.has('Authorization')) {
      headers.set('Authorization', `Bearer ${token}`);
    }
    nextOptions.headers = headers;
  } else {
    const headersInit = Array.isArray(options.headers)
      ? Object.fromEntries(options.headers)
      : { ...(options.headers || {}) };
    if (!('Authorization' in headersInit)) {
      headersInit.Authorization = `Bearer ${token}`;
    }
    nextOptions.headers = headersInit;
  }

  try {
    return await fetch(url, nextOptions);
  } catch (err) {
    if (await shouldTryNodeFetch(err)) {
      return nodeFetch(target, nextOptions);
    }
    throw err;
  }
}

async function shouldTryNodeFetch(err) {
  if (!err) return false;
  const codes = new Set();
  if (err.code) codes.add(err.code);
  if (err.cause?.code) codes.add(err.cause.code);
  if (Array.isArray(err.errors)) {
    for (const nested of err.errors) {
      if (nested?.code) codes.add(nested.code);
    }
  }
  if (codes.has('ENETUNREACH') || codes.has('ECONNREFUSED')) {
    return await canUseNodeFetch();
  }
  const message = typeof err.message === 'string' ? err.message : '';
  const causeMessage = typeof err.cause?.message === 'string' ? err.cause.message : '';
  const combined = `${message} ${causeMessage}`;
  if (combined.includes('ENETUNREACH') || combined.includes('proxy') || combined.includes('fetch failed')) {
    return await canUseNodeFetch();
  }
  return false;
}

let childProcessModulePromise = null;

async function canUseNodeFetch() {
  if (typeof process === 'undefined' || !process?.versions?.node) return false;
  try {
    await loadChildProcessModule();
    return true;
  } catch (err) {
    console.warn('[Hub] Failed to prepare Node HTTPS fallback', err);
    return false;
  }
}

async function loadChildProcessModule() {
  if (!childProcessModulePromise) {
    childProcessModulePromise = import('node:child_process');
  }
  return childProcessModulePromise;
}

async function nodeFetch(url, options = {}) {
  const childProcessMod = await loadChildProcessModule();
  const target = typeof url === 'string' ? url : url.toString();
  const headers = new Headers(options.headers || {});
  const headerArgs = [];
  for (const [key, value] of headers.entries()) {
    headerArgs.push('-H', `${key}: ${value}`);
  }

  const args = ['-L', '--silent', '--show-error', '--fail'];
  args.push(...headerArgs);
  args.push(target);

  return new Promise((resolve, reject) => {
    const child = childProcessMod.spawn('curl', args, {
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    const stderrChunks = [];
    const stdoutChunks = [];

    child.stdout.on('data', (chunk) => stdoutChunks.push(chunk));
    child.stderr.on('data', (chunk) => stderrChunks.push(chunk));
    child.on('error', reject);
    child.on('close', (code) => {
      if (code !== 0) {
        const stderr = Buffer.concat(stderrChunks).toString() || `curl exited with code ${code}`;
        reject(new Error(stderr));
        return;
      }

      const bodyBuffer = Buffer.concat(stdoutChunks);
      const response = new Response(bodyBuffer, {
        status: 200,
        headers: new Headers(),
      });
      resolve(response);
    });
  });
}

async function listRepoFiles(repoId, revision = 'main') {
  const cacheKey = `${repoId}@${revision}`;
  if (repoFileCache.has(cacheKey)) return repoFileCache.get(cacheKey);

  const url = `https://huggingface.co/api/models/${repoId}?revision=${revision}`;
  try {
    const resp = await fetchWithAuth(url);
    if (!resp.ok) throw new Error(`Failed to list repo files: ${resp.status}`);
    const json = await resp.json();
    const files = json.siblings?.map(s => s.rfilename) || [];
    repoFileCache.set(cacheKey, files);
    return files;
  } catch (err) {
    console.warn('[Hub] Could not fetch repo file list – falling back to optimistic fetch', err);
    // Return empty list so caller behaves like old code (may attempt fetch and catch 404)
    repoFileCache.set(cacheKey, []);
    return [];
  }
}

function getDb() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, 1);
      request.onerror = () => reject("Error opening IndexedDB");
      request.onsuccess = () => resolve(request.result);
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME);
        }
      };
    });
  }
  return dbPromise;
}

async function getFileFromDb(key) {
  const db = await getDb();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([STORE_NAME], 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.get(key);
    request.onerror = () => reject("Error reading from DB");
    request.onsuccess = () => resolve(request.result);
  });
}

async function saveFileToDb(key, blob) {
    const db = await getDb();
    return new Promise((resolve, reject) => {
        const transaction = db.transaction([STORE_NAME], 'readwrite');
        const store = transaction.objectStore(STORE_NAME);
        const request = store.put(blob, key);
        request.onerror = () => reject("Error writing to DB");
        request.onsuccess = () => resolve(request.result);
    });
}

/**
 * Download a file from HuggingFace Hub with caching support.
 * @param {string} repoId Model repo ID (e.g., 'nvidia/parakeet-tdt-1.1b')
 * @param {string} filename File to download (e.g., 'encoder-model.onnx')
 * @param {Object} [options]
 * @param {string} [options.revision='main'] Git revision
 * @param {string} [options.subfolder=''] Subfolder within repo
 * @param {Function} [options.progress] Progress callback
 * @returns {Promise<string>} URL to cached file (blob URL)
 */
export async function getModelFile(repoId, filename, options = {}) {
  const { revision = 'main', subfolder = '', progress } = options;
  
  // Construct HF URL
  const baseUrl = 'https://huggingface.co';
  const pathParts = [repoId, 'resolve', revision];
  if (subfolder) pathParts.push(subfolder);
  pathParts.push(filename);
  const url = `${baseUrl}/${pathParts.join('/')}`;
  
  // Check IndexedDB first
  const cacheKey = `hf-${repoId}-${revision}-${subfolder}-${filename}`;
  
  if (typeof indexedDB !== 'undefined') {
    try {
      const cachedBlob = await getFileFromDb(cacheKey);
      if (cachedBlob) {
        console.log(`[Hub] Using cached ${filename} from IndexedDB`);
        return URL.createObjectURL(cachedBlob);
      }
    } catch (e) {
      console.warn('[Hub] IndexedDB cache check failed:', e);
    }
  }
  
  // Download from HF
  console.log(`[Hub] Downloading ${filename} from ${repoId}...`);
  const response = await fetchWithAuth(url);
  if (!response.ok) {
    throw new Error(`Failed to download ${filename}: ${response.status} ${response.statusText}`);
  }
  
  // Stream with progress
  const contentLength = response.headers.get('content-length');
  const total = contentLength ? parseInt(contentLength) : 0;
  let loaded = 0;
  
  const reader = response.body.getReader();
  const chunks = [];
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    chunks.push(value);
    loaded += value.length;
    
    if (progress && total > 0) {
      progress({ loaded, total, file: filename });
    }
  }
  
  // Reconstruct blob
  const blob = new Blob(chunks, { type: response.headers.get('content-type') || 'application/octet-stream' });
  
  // Cache the blob in IndexedDB
  if (typeof indexedDB !== 'undefined') {
    try {
      await saveFileToDb(cacheKey, blob);
       console.log(`[Hub] Cached ${filename} in IndexedDB`);
    } catch (e) {
      console.warn('[Hub] Failed to cache in IndexedDB:', e);
    }
  }
  
  return URL.createObjectURL(blob);
}

/**
 * Download text file from HF Hub.
 * @param {string} repoId Model repo ID
 * @param {string} filename Text file to download
 * @param {Object} [options] Same as getModelFile
 * @returns {Promise<string>} File content as text
 */
export async function getModelText(repoId, filename, options = {}) {
  const blobUrl = await getModelFile(repoId, filename, options);
  const response = await fetchWithAuth(blobUrl);
  const text = await response.text();
  URL.revokeObjectURL(blobUrl); // Clean up blob URL
  return text;
}

/**
 * Convenience function to get all Parakeet model files for a given architecture.
 * @param {string} repoId HF repo (e.g., 'nvidia/parakeet-tdt-1.1b')
 * @param {Object} [options]
 * @param {('int8'|'fp32')} [options.encoderQuant='int8'] Encoder quantization
 * @param {('int8'|'fp32')} [options.decoderQuant='int8'] Decoder quantization
 * @param {('nemo80'|'nemo128')} [options.preprocessor='nemo128'] Preprocessor variant
 * @param {('webgpu'|'wasm')} [options.backend='webgpu'] Backend to use
 * @param {Function} [options.progress] Progress callback
 * @returns {Promise<{urls: object, filenames: object}>}
 */
export async function getParakeetModel(repoId, options = {}) {
  const { encoderQuant = 'int8', decoderQuant = 'int8', preprocessor = 'nemo128', backend = 'webgpu', progress } = options;
  
  // Decide quantisation per component
  let encoderQ = encoderQuant;
  let decoderQ = decoderQuant;

  if (backend.startsWith('webgpu') && encoderQ === 'int8') {
    console.warn('[Hub] Forcing encoder to fp32 on WebGPU (int8 unsupported)');
    encoderQ = 'fp32';
  }

  const encoderSuffix = encoderQ === 'int8' ? '.int8.onnx' : '.onnx';
  const decoderSuffix = decoderQ === 'int8' ? '.int8.onnx' : '.onnx';

  const encoderName = `encoder-model${encoderSuffix}`;
  const decoderName = `decoder_joint-model${decoderSuffix}`;

  const repoFiles = await listRepoFiles(repoId, options.revision || 'main');

  const filesToGet = [
    { key: 'encoderUrl', name: encoderName },
    { key: 'decoderUrl', name: decoderName },
    { key: 'tokenizerUrl', name: 'vocab.txt' },
    { key: 'preprocessorUrl', name: `${preprocessor}.onnx` },
  ];

  // Conditionally include external data files only if they exist in the repo file list.
  if (repoFiles.includes(`${encoderName}.data`)) {
    filesToGet.push({ key: 'encoderDataUrl', name: `${encoderName}.data` });
  }

  if (repoFiles.includes(`${decoderName}.data`)) {
    filesToGet.push({ key: 'decoderDataUrl', name: `${decoderName}.data` });
  }

  const results = {
      urls: {},
      filenames: {
          encoder: encoderName,
          decoder: decoderName
      },
      quantisation: { encoder: encoderQ, decoder: decoderQ }
  };
  
  for (const { key, name } of filesToGet) {
    try {
        const wrappedProgress = progress ? (p) => progress({ ...p, file: name }) : undefined;
        results.urls[key] = await getModelFile(repoId, name, { ...options, progress: wrappedProgress });
    } catch (e) {
        if (key.endsWith('DataUrl')) {
            console.warn(`[Hub] Optional external data file not found: ${name}. This is expected if the model is small.`);
            results.urls[key] = null;
        } else {
            throw e;
        }
    }
  }
  
  return results;
}
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Agent, ProxyAgent, setGlobalDispatcher } from 'undici';

if (typeof process !== 'undefined' && process?.env) {
  const proxyUrl = process.env.HTTPS_PROXY || process.env.HTTP_PROXY;
  if (proxyUrl) {
    setGlobalDispatcher(new ProxyAgent(proxyUrl));
  } else {
    setGlobalDispatcher(new Agent());
  }
} else {
  setGlobalDispatcher(new Agent());
}

const globalScope = globalThis;

if (!globalScope.self) {
  globalScope.self = globalScope;
}

if (!globalScope.window) {
  globalScope.window = globalScope;
}

if (!globalScope.navigator) {
  globalScope.navigator = {};
}

if (globalScope.navigator.hardwareConcurrency === undefined) {
  globalScope.navigator.hardwareConcurrency = 4;
}

if (globalScope.navigator.userAgent === undefined) {
  globalScope.navigator.userAgent = 'node';
}

if (globalScope.navigator.language === undefined) {
  globalScope.navigator.language = 'en-US';
}

if (!globalScope.location) {
  globalScope.location = new URL('https://localhost/');
}

// Force single-threaded WASM to avoid worker requirements in Node tests.
if (typeof globalScope.SharedArrayBuffer !== 'undefined') {
  try {
    Object.defineProperty(globalScope, 'SharedArrayBuffer', {
      value: undefined,
      configurable: true,
    });
  } catch {
    globalScope.SharedArrayBuffer = undefined;
  }
}

if (!globalScope.Blob) {
  throw new Error('Global Blob constructor not available');
}

const blobStore = new Map();
let blobSeq = 0;

URL.createObjectURL = (blob) => {
  const id = `blob:parakeet-test-${++blobSeq}`;
  blobStore.set(id, blob);
  return id;
};

URL.revokeObjectURL = (url) => {
  blobStore.delete(url);
};

const originalFsReadFile = fs.promises.readFile.bind(fs.promises);

async function readBlobFile(path, options) {
  if (typeof path === 'string' && path.startsWith('blob:parakeet-test-')) {
    const blob = blobStore.get(path);
    if (!blob) {
      throw new Error(`No blob found for URL ${path}`);
    }
    const buffer = Buffer.from(await blob.arrayBuffer());
    if (options && typeof options === 'object' && 'encoding' in options && options.encoding) {
      return buffer.toString(options.encoding);
    }
    if (typeof options === 'string') {
      return buffer.toString(options);
    }
    return buffer;
  }
  return null;
}

fs.promises.readFile = async (path, options) => {
  const blobResult = await readBlobFile(path, options);
  if (blobResult !== null) return blobResult;
  return originalFsReadFile(path, options);
};

const originalFetch = globalScope.fetch.bind(globalScope);

function isRequestLike(obj) {
  return typeof obj === 'object' && obj !== null && typeof obj.url === 'string';
}

globalScope.fetch = async function patchedFetch(resource, init) {
  let url = null;
  if (typeof resource === 'string') {
    url = resource;
  } else if (isRequestLike(resource)) {
    url = resource.url;
  }

  if (typeof url === 'string') {
    if (url.startsWith('blob:parakeet-test-')) {
      const blob = blobStore.get(url);
      if (!blob) {
        throw new Error(`No blob found for URL ${url}`);
      }
      return new Response(blob.stream(), {
        headers: { 'content-type': blob.type || 'application/octet-stream' },
      });
    }

    if (url.startsWith('file://')) {
      const path = fileURLToPath(url);
      const data = await fs.promises.readFile(path);
      return new Response(data);
    }
  }

  return originalFetch(resource, init);
};

/**
 * Browser-based WebGPU test using Playwright and onnxruntime-web
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFileSync, readdirSync } from 'node:fs';
import { join, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const projectRoot = join(__dirname, '..');
const localOrtPath = '/data/onnxruntime/js/web/dist';

// Simple static file server that serves from both project root and local ORT
function startServer(port) {
  const mimeTypes = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.mjs': 'application/javascript',
    '.json': 'application/json',
    '.wasm': 'application/wasm',
    '.onnx': 'application/octet-stream',
  };

  const server = createServer((req, res) => {
    let filePath;
    const url = req.url.split('?')[0]; // Remove query params

    // Serve local ORT files from /ort-local/ path
    if (url.startsWith('/ort-local/')) {
      filePath = join(localOrtPath, url.replace('/ort-local/', ''));
    } else {
      filePath = join(projectRoot, url === '/' ? '/test-webgpu-browser.html' : url);
    }

    try {
      const content = readFileSync(filePath);
      const ext = extname(filePath);
      res.writeHead(200, {
        'Content-Type': mimeTypes[ext] || 'application/octet-stream',
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      });
      res.end(content);
    } catch (e) {
      console.error(`[Server] 404: ${url} -> ${filePath}`);
      res.writeHead(404);
      res.end('Not found: ' + url);
    }
  });

  return new Promise((resolve) => {
    server.listen(port, () => resolve(server));
  });
}

test('asr1 WebGPU browser test with onnxruntime-web', { timeout: 300_000 }, async (t) => {
  const port = 8099;
  const server = await startServer(port);

  let browser;
  try {
    console.log('[Test] Launching Chrome with WebGPU flags...');

    // Launch Chrome with WebGPU enabled
    browser = await chromium.launch({
      headless: true,
      args: [
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan,UseSkiaRenderer',
        '--use-vulkan',
        '--disable-vulkan-surface',
        '--enable-gpu-rasterization',
        '--ignore-gpu-blocklist',
        '--use-angle=vulkan',
      ],
    });

    const context = await browser.newContext();
    const page = await context.newPage();

    // Capture console logs
    const logs = [];
    page.on('console', msg => {
      const text = msg.text();
      logs.push(text);
      console.log(`[Browser] ${text}`);
    });

    page.on('pageerror', err => {
      console.error(`[Browser Error] ${err.message}`);
    });

    // Navigate to test page
    console.log(`[Test] Loading http://localhost:${port}/test-webgpu-browser.html`);
    await page.goto(`http://localhost:${port}/test-webgpu-browser.html`);

    // Check WebGPU availability first
    const hasWebGPU = await page.evaluate(async () => {
      if (!navigator.gpu) return { available: false, reason: 'navigator.gpu not found' };
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return { available: false, reason: 'No adapter found' };
        return { available: true, adapter: adapter.info?.device || 'unknown' };
      } catch (e) {
        return { available: false, reason: e.message };
      }
    });

    console.log('[Test] WebGPU status:', hasWebGPU);

    if (!hasWebGPU.available) {
      console.log('[Test] WebGPU not available, testing WASM fallback...');
    }

    // Run the test with webgpu-hybrid backend
    const testBackend = hasWebGPU.available ? 'webgpu-hybrid' : 'wasm';
    console.log(`[Test] Running test with backend: ${testBackend}`);

    // Call the test function
    await page.evaluate((backend) => window.runTest(backend), testBackend);

    // Wait for result (poll window.testResult)
    console.log('[Test] Waiting for test to complete...');
    const result = await page.waitForFunction(
      () => window.testResult && window.testResult.status !== 'running',
      { timeout: 180000 }
    ).then(() => page.evaluate(() => window.testResult));

    const passed = result.status === 'passed';
    console.log('[Test] Result:', passed ? 'SUCCESS' : 'FAILED');
    console.log('[Test] Backend used:', result.backend);

    if (!passed) {
      console.error('[Test] Error:', result.error);
      console.error('[Test] Stack:', result.stack);
    } else {
      console.log('[Test] Transcription time:', result.elapsed, 'ms');
    }

    assert.ok(passed, `Test failed: ${result.error}`);

  } finally {
    if (browser) await browser.close();
    server.close();
  }
});

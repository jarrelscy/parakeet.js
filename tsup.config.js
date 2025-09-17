import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.js'],
  format: ['esm', 'cjs'],
  target: 'es2020',
  splitting: false,
  sourcemap: true,
  clean: true,
  dts: false,
  minify: false,
  platform: 'browser',
  globalName: 'Parakeet',
  noExternal: ['onnxruntime-web'],
});

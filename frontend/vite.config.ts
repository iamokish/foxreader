/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import preact from '@preact/preset-vite';
import path from 'path';

export default defineConfig({
  plugins: [preact()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/__tests__/setup.ts'],
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
  },
  build: {
    outDir: 'static',
    emptyOutDir: false,
    // Emit bundled-dependency licenses (see NOTICE §1B): writes
    // static/.vite/license.md listing every package bundled into app.js.
    // embed_assets.py embeds the whole static/ dir, so the license file
    // ships inside the binary alongside the bundle. (Note: top-level
    // esbuild.legalComments does NOT work here — Vite 8 minifies with oxc
    // and ignores esbuild options, as the build warning confirms.)
    license: true,
    rollupOptions: {
      input: 'src/main.tsx',
      output: {
        entryFileNames: 'app.js',
        format: 'iife',
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  define: {
    'import.meta': {},
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:7954',
      '/img_serve': 'http://127.0.0.1:7954',
      '/translate': 'http://127.0.0.1:7954',
      '/ml': 'http://127.0.0.1:7954',
      '/inpaint': 'http://127.0.0.1:7954',
      '/ws': {
        target: 'ws://127.0.0.1:7954',
        ws: true,
      },
    },
  },
});
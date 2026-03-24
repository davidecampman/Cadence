import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Detect if we're running inside Tauri
const isTauri = process.env.TAURI_ENV_PLATFORM !== undefined;

export default defineConfig({
  plugins: [react()],

  // Tauri expects a fixed port during dev
  server: {
    port: 5173,
    strictPort: true,
    proxy: isTauri
      ? undefined // Tauri dev: frontend talks directly to localhost:8000
      : {
          '/api': 'http://localhost:8000',
          '/ws': {
            target: 'ws://localhost:8000',
            ws: true,
          },
        },
  },

  // Prevent Vite from obscuring Rust errors in dev
  clearScreen: false,

  // Env prefix for Tauri environment variables
  envPrefix: ['VITE_', 'TAURI_ENV_'],

  build: {
    // Tauri uses Chromium on Windows and WebKit on macOS/Linux
    target: isTauri ? 'esnext' : 'modules',
    // Tauri will minify in release builds
    minify: !isTauri ? 'esbuild' : false,
    sourcemap: !!isTauri,
  },
})

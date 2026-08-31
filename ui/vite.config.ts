import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base: './' emits relative asset URLs so the bundle works under the
// /dashboard/ path prefix the Go server mounts it at. outDir stays the
// default (ui/dist); the maintainer wires the final embed path at integration.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    outDir: 'dist',
    // The whole app is embedded in a Go binary; keep the bundle honest.
    chunkSizeWarningLimit: 700,
  },
  server: {
    port: 5273,
    proxy: {
      // Dev convenience: point the API host with ROSTAM_API (default loopback).
      '/v1': process.env.ROSTAM_API || 'http://127.0.0.1:8080',
      '/metrics': process.env.ROSTAM_API || 'http://127.0.0.1:8080',
    },
  },
});

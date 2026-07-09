import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    // bind IPv4 explicitly — backend listens on 127.0.0.1 and the ws proxy
    // breaks when vite resolves localhost to ::1
    host: '127.0.0.1',
    port: 6130,
    strictPort: true,
    proxy: {
      '/api/v1': {
        target: process.env.API_PROXY_TARGET || 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: (process.env.API_PROXY_TARGET || 'http://localhost:8000').replace('http', 'ws'),
        ws: true,
      },
    },
  },
})

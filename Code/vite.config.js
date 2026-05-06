import { defineConfig } from 'vite';
import { existsSync, readFileSync } from 'fs';
import { join } from 'path';
import basicSsl from '@vitejs/plugin-basic-ssl';

function ortWasmPlugin() {
  const distDir = join(process.cwd(), 'node_modules', 'onnxruntime-web', 'dist');
  return {
    name: 'ort-wasm',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const match = req.url?.match(/^\/(ort-wasm[^?/]*\.wasm)/);
        if (match) {
          const wasmPath = join(distDir, match[1]);
          if (existsSync(wasmPath)) {
            res.setHeader('Content-Type', 'application/wasm');
            res.end(readFileSync(wasmPath));
            return;
          }
        }
        next();
      });
    },
  };
}

const isDev = process.env.NODE_ENV !== 'production';

export default defineConfig({
  plugins: [ortWasmPlugin(), ...(isDev ? [basicSsl()] : [])],
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  build: {
    target: 'es2020',
  },
  server: {
    host: true,
  },
});

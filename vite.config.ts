import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    lib: {
      entry: './src/similarity.ts',
      name: 'similarity',
      fileName: 'similarity',
      formats: ['es']
    },
    rollupOptions: {
      output: {
        dir: './dist'
      }
    }
  },
  test: {
    globals: true,
    environment: 'jsdom',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html']
    }
  }
});

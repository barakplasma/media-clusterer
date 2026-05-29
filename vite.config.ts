import { readFileSync } from 'fs';
import { defineConfig } from 'vite';
import { execSync } from 'child_process';

const getGitInfo = (cmd: string, envVar?: string) => {
  if (envVar && process.env[envVar]) return process.env[envVar];
  try {
    return execSync(cmd).toString().trim();
  } catch (e) {
    return 'unknown';
  }
};

const gitBranch = getGitInfo('git rev-parse --abbrev-ref HEAD', 'GITHUB_HEAD_REF') || getGitInfo('git rev-parse --abbrev-ref HEAD', 'GITHUB_REF_NAME');
const gitCommit = getGitInfo('git rev-parse --short HEAD', 'GITHUB_SHA')?.substring(0, 7);
const appVersion = JSON.parse(readFileSync('./package.json', 'utf-8')).version as string;

// onnxruntime-web WASM files are loaded at runtime from the CDN via
// ort.env.wasm.wasmPaths (set in src/sapiens2.ts). Bundling them into dist
// is both wasteful and hits Cloudflare Pages' 25 MiB per-file limit.
const excludeOrtWasm = {
  name: 'exclude-ort-wasm',
  generateBundle(_: unknown, bundle: Record<string, { fileName?: string }>) {
    for (const key of Object.keys(bundle)) {
      if (bundle[key].fileName?.includes('ort-wasm')) {
        delete bundle[key];
      }
    }
  },
};

export default defineConfig({
  define: {
    __GIT_BRANCH__: JSON.stringify(gitBranch),
    __GIT_COMMIT__: JSON.stringify(gitCommit),
    __APP_VERSION__: JSON.stringify(appVersion),
  },
  plugins: [excludeOrtWasm],
  build: {
    sourcemap: true,
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

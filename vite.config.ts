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

export default defineConfig({
  define: {
    __GIT_BRANCH__: JSON.stringify(gitBranch),
    __GIT_COMMIT__: JSON.stringify(gitCommit),
  },
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

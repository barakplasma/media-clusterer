import { defineConfig } from 'vite';
import { execSync } from 'child_process';

const gitBranch = execSync('git rev-parse --abbrev-ref HEAD').toString().trim();
const gitCommit = execSync('git rev-parse --short HEAD').toString().trim();

export default defineConfig({
  define: {
    __GIT_BRANCH__: JSON.stringify(gitBranch),
    __GIT_COMMIT__: JSON.stringify(gitCommit),
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

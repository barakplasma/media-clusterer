import { defineConfig } from '@playwright/test'

// Real-GPU end-to-end tests. CI runs them on GitHub's macOS runners, whose
// Chrome gets the Apple GPU through WebGPU. SWIFTSHADER=<chrome path> runs
// them on a machine without a GPU (slow; local debugging only).
const swiftshader = process.env.SWIFTSHADER
const PIXEL_UA =
  'Mozilla/5.0 (Linux; Android 16; Pixel 10) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Mobile Safari/537.36'

export default defineConfig({
  testDir: 'e2e',
  timeout: 40 * 60_000,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: 'http://127.0.0.1:5173',
    ...(swiftshader
      ? {
          launchOptions: {
            executablePath: swiftshader,
            args: ['--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--use-webgpu-adapter=swiftshader'],
          },
        }
      : { channel: 'chrome' }),
  },
  webServer: {
    command: 'npx vite --host 127.0.0.1 --port 5173 --strictPort',
    url: 'http://127.0.0.1:5173',
    reuseExistingServer: !process.env.CI,
  },
  // e2e/safari runs under node:test with safaridriver, not here.
  projects: [
    // The app picks its phone profile (fewer image tokens, fixed sequence length) from the UA.
    { name: 'phone', testMatch: 'embeddinggemma2.spec.ts', use: { userAgent: PIXEL_UA } },
    { name: 'desktop', testMatch: 'embeddinggemma2.spec.ts' },
    // The WebNN path (src/vlmWebnn.ts), which the app prefers whenever the
    // browser has WebNN; the flag below turns it on. No GPU needed.
    {
      name: 'webnn',
      testMatch: 'embeddinggemma2.spec.ts',
      use: {
        userAgent: PIXEL_UA,
        launchOptions: {
          ...(swiftshader ? { executablePath: swiftshader } : {}),
          args: ['--enable-features=WebMachineLearningNeuralNetwork'],
        },
      },
    },
    // No GPU needed: `npm test`, the pre-commit hook and the Linux deploy job.
    { name: 'cpu', testMatch: 'cpu.spec.ts', timeout: 60_000 },
    // Chrome's built-in Gemini Nano; the spec launches its own persistent
    // Chrome profile, since the Prompt API is off in Playwright's default
    // (incognito-like) contexts.
    { name: 'chrome-ai', testMatch: 'chrome-ai.spec.ts', timeout: 25 * 60_000 },
  ],
})

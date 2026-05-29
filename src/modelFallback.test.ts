import { describe, it, expect, vi } from 'vitest';

// modelFallback -> sapiens2 -> onnxruntime-web. Stub the native dep so the
// module graph loads in jsdom without pulling in the real runtime.
vi.mock('onnxruntime-web', () => ({
  env: { wasm: {}, webgpu: {} },
  InferenceSession: { create: vi.fn() },
  Tensor: vi.fn(),
}));

import {
  normalizeHost,
  modelDownloadUrls,
  buildUploadCache,
  isDownloadError,
  HF_HOST,
} from './modelFallback';

describe('normalizeHost', () => {
  it('trims whitespace and trailing slashes', () => {
    expect(normalizeHost('  https://h.co/  ')).toBe('https://h.co');
    expect(normalizeHost('https://h.co///')).toBe('https://h.co');
  });
  it('returns empty string for blank/nullish input', () => {
    expect(normalizeHost('')).toBe('');
    expect(normalizeHost('   ')).toBe('');
    expect(normalizeHost(undefined)).toBe('');
    expect(normalizeHost(null)).toBe('');
  });
});

describe('modelDownloadUrls', () => {
  it('returns the single onnx URL for a sapiens2 variant', () => {
    const urls = modelDownloadUrls('sapiens2-int8');
    expect(urls).toHaveLength(1);
    expect(urls[0]).toBe(`${HF_HOST}/barakplasma/sapiens2-onnx/resolve/main/sapiens2_0.1b_int8.onnx`);
  });

  it('lists the vision repo files for nomic', () => {
    const urls = modelDownloadUrls('nomic');
    expect(urls.some((u) => u.endsWith('/nomic-embed-vision-v1.5/resolve/main/config.json'))).toBe(true);
    expect(urls.some((u) => u.endsWith('/onnx/model.onnx'))).toBe(true);
    expect(urls.every((u) => u.startsWith(HF_HOST))).toBe(true);
  });

  it('appends text repo files when includeText is set', () => {
    const urls = modelDownloadUrls('nomic', { includeText: true });
    expect(urls.some((u) => u.includes('/nomic-embed-text-v1.5/'))).toBe(true);
    expect(urls.some((u) => u.endsWith('/tokenizer.json'))).toBe(true);
  });

  it('returns the text repo for chrome-ai', () => {
    const urls = modelDownloadUrls('chrome-ai');
    expect(urls.every((u) => u.includes('/nomic-embed-text-v1.5/'))).toBe(true);
  });

  it('honors a custom host', () => {
    const urls = modelDownloadUrls('nomic', { host: 'https://mirror.corp/' });
    expect(urls.every((u) => u.startsWith('https://mirror.corp/'))).toBe(true);
  });
});

describe('buildUploadCache', () => {
  const fileFor = (name: string) =>
    new File([new Uint8Array([1, 2, 3])], name.split('/').pop() ?? name);

  it('matches a request by path after /resolve/main/', async () => {
    const files = new Map<string, File>([
      ['nomic-embed-vision-v1.5/config.json', fileFor('config.json')],
      ['nomic-embed-vision-v1.5/onnx/model.onnx', fileFor('model.onnx')],
    ]);
    const cache = buildUploadCache(files);

    const hit = await cache.match(
      'https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5/resolve/main/onnx/model.onnx',
    );
    expect(hit).toBeInstanceOf(Response);

    const cfg = await cache.match(
      'https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5/resolve/main/config.json',
    );
    expect(cfg).toBeInstanceOf(Response);
  });

  it('returns undefined for files that were not uploaded', async () => {
    const cache = buildUploadCache(new Map([['repo/config.json', fileFor('config.json')]]));
    const miss = await cache.match(
      'https://huggingface.co/x/y/resolve/main/onnx/model.onnx',
    );
    expect(miss).toBeUndefined();
  });

  it('put is a no-op that resolves', async () => {
    const cache = buildUploadCache(new Map());
    await expect(cache.put()).resolves.toBeUndefined();
  });
});

describe('isDownloadError', () => {
  it('matches the Chrome/Firefox fetch network message', () => {
    expect(isDownloadError(new TypeError('Failed to fetch'))).toBe(true);
  });
  it('matches the Safari fetch network message', () => {
    expect(isDownloadError(new TypeError('Load failed'))).toBe(true);
  });
  it('matches our sapiens2 download error message', () => {
    expect(isDownloadError(new Error('Sapiens2 download failed: 404'))).toBe(true);
  });
  it('matches transformers "could not locate" errors', () => {
    expect(isDownloadError(new Error('Could not locate file: config.json'))).toBe(true);
  });
  it('ignores AbortError', () => {
    const e = new Error('aborted');
    e.name = 'AbortError';
    expect(isDownloadError(e)).toBe(false);
  });
  it('does not flag generic compute errors', () => {
    expect(isDownloadError(new Error('WebGPU device lost'))).toBe(false);
  });
  it('does not flag a bare TypeError from a programming bug', () => {
    // e.g. "x is not a function" / "Cannot read properties of undefined"
    expect(isDownloadError(new TypeError('foo.bar is not a function'))).toBe(false);
    expect(isDownloadError(new TypeError("Cannot read properties of undefined (reading 'run')"))).toBe(false);
  });
});

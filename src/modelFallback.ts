/**
 * Offline / corporate-proxy fallback for model downloads.
 *
 * When a model can't be fetched from huggingface.co (the user is offline or
 * behind a proxy/firewall that blocks the Hub), the app surfaces a modal that
 * offers two recovery paths:
 *   A) upload the model file(s) from local disk, and
 *   B) point at an alternative HuggingFace-compatible host (a corporate mirror
 *      such as Artifactory).
 *
 * This module holds the pure, unit-testable pieces: host/URL helpers and the
 * Transformers.js custom-cache builder. The modal wiring lives in app.ts where
 * it has access to the DOM and settings.
 */

import { sapiens2Url, type Sapiens2Variant } from './sapiens2';

export const HF_HOST = 'https://huggingface.co';

// Files each Transformers.js repo fetches at fp32. Used to show the user exactly
// what to download and to match locally-uploaded files against request URLs.
const NOMIC_VISION_REPO = 'nomic-ai/nomic-embed-vision-v1.5';
const NOMIC_TEXT_REPO = 'nomic-ai/nomic-embed-text-v1.5';
const VISION_FILES = ['config.json', 'preprocessor_config.json', 'onnx/model.onnx'];
const TEXT_FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'onnx/model.onnx'];

/** Trim and strip trailing slashes from a user-entered host. '' if blank. */
export function normalizeHost(host: string | undefined | null): string {
  return (host ?? '').trim().replace(/\/+$/, '');
}

function repoUrls(repo: string, files: string[], host: string): string[] {
  const base = normalizeHost(host) || HF_HOST;
  return files.map((f) => `${base}/${repo}/resolve/main/${f}`);
}

/**
 * Full download URLs required for a model variant. Shown in the fallback modal
 * (always against the canonical Hub host so the links are copy/curl-friendly).
 * `includeText` adds the text-embedding repo (used when text search is enabled).
 */
export function modelDownloadUrls(
  variant: string,
  opts: { host?: string; includeText?: boolean } = {},
): string[] {
  const host = opts.host ?? HF_HOST;
  if (variant.startsWith('sapiens2')) {
    const v = (variant.split('-')[1] ?? 'fp16') as Sapiens2Variant;
    return [sapiens2Url(v, host)];
  }
  if (variant === 'chrome-ai') {
    // Chrome AI itself is browser-managed; only the text model is downloaded.
    return repoUrls(NOMIC_TEXT_REPO, TEXT_FILES, host);
  }
  const urls = repoUrls(NOMIC_VISION_REPO, VISION_FILES, host);
  if (opts.includeText) urls.push(...repoUrls(NOMIC_TEXT_REPO, TEXT_FILES, host));
  return urls;
}

/**
 * Build a Web Cache-compatible object that serves locally-uploaded model files
 * to Transformers.js. Assign it to `env.customCache` with `env.useCustomCache`.
 *
 * Files are keyed by their relative path (from a `webkitdirectory` upload, that
 * includes the top folder, e.g. `nomic-embed-vision-v1.5/onnx/model.onnx`).
 * `match` resolves a request URL to an uploaded file by comparing the path that
 * follows `/resolve/<rev>/`, falling back to the basename. Returning `undefined`
 * lets Transformers.js fall through to a normal fetch for anything not uploaded.
 */
export function buildUploadCache(files: Map<string, File>): {
  match(request: RequestInfo | URL): Promise<Response | undefined>;
  put(): Promise<void>;
} {
  const entries = [...files.entries()].map(([rel, file]) => [rel.replace(/^\/+/, ''), file] as const);

  const wantedPath = (url: string): string =>
    url.replace(/^.*\/resolve\/[^/]+\//, '').replace(/[?#].*$/, '').replace(/^\/+/, '');

  return {
    async match(request: RequestInfo | URL): Promise<Response | undefined> {
      const url =
        typeof request === 'string' ? request : request instanceof URL ? request.href : request.url;
      const want = wantedPath(url);
      const base = want.split('/').pop() ?? want;

      let hit: File | undefined;
      for (const [rel, file] of entries) {
        if (rel === want || rel.endsWith('/' + want) || rel === base || rel.endsWith('/' + base)) {
          hit = file;
          break;
        }
      }
      if (!hit) return undefined;
      return new Response(await hit.arrayBuffer(), {
        status: 200,
        headers: { 'Content-Type': 'application/octet-stream' },
      });
    },
    async put(): Promise<void> {
      /* no-op: this cache only serves uploaded files, it doesn't persist them */
    },
  };
}

/** Heuristic: did this error come from a failed model *download* (vs. compute)? */
export function isDownloadError(err: unknown): boolean {
  if (!err) return false;
  const e = err as Error;
  if (e.name === 'AbortError') return false;
  if (e.name === 'TypeError') return true; // fetch() network failure
  if (typeof navigator !== 'undefined' && navigator.onLine === false) return true;
  const m = (e.message || '').toLowerCase();
  return /download failed|failed to fetch|network ?error|could not locate|unauthorized|err_|http error|status (4|5)\d\d|getaddrinfo|enotfound|\b(403|404|429|500|502|503|504)\b/.test(
    m,
  );
}

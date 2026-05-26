/**
 * Chrome built-in AI Prompt API integration.
 * Stage 1 of the chrome-ai embedding pipeline: image → text description.
 * Stage 2 (description text → 768-dim vector) is handled in app.ts via textExtractor.
 *
 * Requires Chrome 138+ with:
 *   chrome://flags/#optimization-guide-on-device-model  → Enabled BypassPerfRequirement
 *   chrome://flags/#prompt-api-for-gemini-nano           → Enabled
 */

// ── Type declarations for Chrome's built-in LanguageModel API ────────────────

export type LanguageModelAvailability =
  | 'available'
  | 'downloadable'
  | 'downloading'
  | 'unavailable';

export interface LanguageModelSession {
  prompt(
    inputs: Array<{ type: 'text'; value: string } | { type: 'image'; value: ImageBitmap | Blob }>,
    options?: { signal?: AbortSignal }
  ): Promise<string>;
  destroy(): void;
}

interface LanguageModelStatic {
  availability(options?: {
    expectedInputs?: Array<{ type: string }>;
  }): Promise<LanguageModelAvailability>;
  create(options?: {
    expectedInputs?: Array<{ type: string }>;
    signal?: AbortSignal;
  }): Promise<LanguageModelSession>;
}

declare global {
  const LanguageModel: LanguageModelStatic | undefined;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const DESCRIBE_PROMPT =
  'Describe this image for photo organization and clustering. ' +
  'Include: main subjects, setting or location, dominant colors, mood, ' +
  'lighting conditions, and any activity taking place. ' +
  'Be specific and concise (2-4 sentences).';

// Recycle the session after this many .prompt() calls to bound context growth.
const SESSION_RECYCLE_INTERVAL = 15;

// ── Availability check ────────────────────────────────────────────────────────

export async function getChromeAIAvailability(): Promise<LanguageModelAvailability> {
  // Support both new global form and legacy window.ai.languageModel (pre-138)
  const api = (typeof LanguageModel !== 'undefined' ? LanguageModel : undefined)
    ?? (window as unknown as { ai?: { languageModel?: LanguageModelStatic } }).ai?.languageModel;
  if (!api) return 'unavailable';
  try {
    return await api.availability({
      expectedInputs: [{ type: 'image' }, { type: 'text' }],
    });
  } catch {
    return 'unavailable';
  }
}

function getAPI(): LanguageModelStatic {
  const api = (typeof LanguageModel !== 'undefined' ? LanguageModel : undefined)
    ?? (window as unknown as { ai?: { languageModel?: LanguageModelStatic } }).ai?.languageModel;
  if (!api) throw new Error('Chrome AI Prompt API not available. Enable chrome://flags/#prompt-api-for-gemini-nano in Chrome 138+.');
  return api;
}

// ── Session manager ───────────────────────────────────────────────────────────

/**
 * Manages a LanguageModel session with automatic recycling to bound context growth.
 * Each call to .describe() appends to the session's conversation history;
 * recycling every SESSION_RECYCLE_INTERVAL calls prevents unbounded memory use.
 */
export class ChromeAISessionManager {
  private session: LanguageModelSession | null = null;
  private callCount = 0;

  async describe(image: ImageBitmap | Blob, signal?: AbortSignal): Promise<string> {
    const api = getAPI();

    if (this.session && this.callCount > 0 && this.callCount % SESSION_RECYCLE_INTERVAL === 0) {
      this.session.destroy();
      this.session = null;
    }

    if (!this.session) {
      this.session = await api.create({
        expectedInputs: [{ type: 'image' }, { type: 'text' }],
        signal,
      });
    }

    const description = await this.session.prompt(
      [
        { type: 'image', value: image },
        { type: 'text', value: DESCRIBE_PROMPT },
      ],
      { signal }
    );

    this.callCount++;
    return description.trim();
  }

  destroy(): void {
    if (this.session) {
      this.session.destroy();
      this.session = null;
    }
    this.callCount = 0;
  }
}

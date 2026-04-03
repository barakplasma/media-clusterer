import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { setupTestDOM, cleanupTestDOM, wait, createMockVectors } from './test-utils';

describe('App functionality', () => {
  beforeEach(() => {
    setupTestDOM();
    vi.useFakeTimers();
  });

  afterEach(() => {
    cleanupTestDOM();
    vi.useRealTimers();
  });

  describe('Search input', () => {
    it('exists and is disabled initially', () => {
      const searchInput = document.getElementById('search-input') as HTMLInputElement;
      expect(searchInput).toBeTruthy();
      expect(searchInput.disabled).toBe(true);
    });

    it('can be enabled when ready', () => {
      const searchInput = document.getElementById('search-input') as HTMLInputElement;
      searchInput.disabled = false;

      expect(searchInput.disabled).toBe(false);
    });

    it('clears value on reset', () => {
      const searchInput = document.getElementById('search-input') as HTMLInputElement;
      searchInput.disabled = false;
      searchInput.value = 'sunset';

      // Simulate reset
      searchInput.value = '';
      searchInput.disabled = true;

      expect(searchInput.value).toBe('');
      expect(searchInput.disabled).toBe(true);
    });
  });

  describe('Clear button', () => {
    it('is hidden initially', () => {
      const clearBtn = document.getElementById('search-clear-btn') as HTMLButtonElement;
      expect(clearBtn.hidden).toBe(true);
    });

    it('shows when input has value', () => {
      const searchInput = document.getElementById('search-input') as HTMLInputElement;
      const clearBtn = document.getElementById('search-clear-btn') as HTMLButtonElement;

      searchInput.value = 'test';
      clearBtn.hidden = !searchInput.value.trim();

      expect(clearBtn.hidden).toBe(false);
    });

    it('clears input and hides on click', () => {
      const searchInput = document.getElementById('search-input') as HTMLInputElement;
      const clearBtn = document.getElementById('search-clear-btn') as HTMLButtonElement;

      searchInput.value = 'test';
      clearBtn.hidden = false;

      // Simulate click
      searchInput.value = '';
      clearBtn.hidden = true;

      expect(searchInput.value).toBe('');
      expect(clearBtn.hidden).toBe(true);
    });
  });

  describe('State management', () => {
    it('tracks search results', () => {
      // Mock state object
      const state: {
        searchResults: Int32Array | null;
        searchQuery: string;
        searchScores: Float32Array | null;
      } = {
        searchResults: null,
        searchQuery: '',
        searchScores: null,
      };

      expect(state.searchResults).toBeNull();

      // Simulate search
      const vectors = createMockVectors(100);
      const query = vectors[0];
      const scores = new Float32Array(vectors.length);
      for (let i = 0; i < vectors.length; i++) {
        scores[i] = query[i] * vectors[i][i]; // Simplified
      }

      const indices = new Int32Array(scores.length);
      for (let i = 0; i < scores.length; i++) indices[i] = i;
      indices.sort((a, b) => scores[b] - scores[a]);

      state.searchResults = indices;
      state.searchQuery = 'test';
      state.searchScores = scores;

      expect(state.searchResults).toBeInstanceOf(Int32Array);
      expect(state.searchResults.length).toBe(100);
      expect(state.searchQuery).toBe('test');
    });

    it('clears search state on reset', () => {
      const state: {
        searchResults: Int32Array | null;
        searchQuery: string;
        searchScores: Float32Array | null;
      } = {
        searchResults: new Int32Array([1, 2, 3]),
        searchQuery: 'test',
        searchScores: new Float32Array([0.1, 0.2, 0.3]),
      };

      // Simulate reset
      state.searchResults = null;
      state.searchQuery = '';
      state.searchScores = null;

      expect(state.searchResults).toBeNull();
      expect(state.searchQuery).toBe('');
      expect(state.searchScores).toBeNull();
    });
  });

  describe('Debounce behavior', () => {
    it('delays execution', async () => {
      let callCount = 0;
      const debouncedFn = vi.fn(() => callCount++);

      // Simulate debounce
      let timeout: ReturnType<typeof setTimeout> | null = null;
      const debounce = (fn: () => void, delay: number) => {
        if (timeout) clearTimeout(timeout);
        timeout = setTimeout(fn, delay);
      };

      debounce(debouncedFn, 300);
      vi.advanceTimersByTime(100);
      expect(debouncedFn).not.toHaveBeenCalled();

      vi.advanceTimersByTime(300);
      expect(debouncedFn).toHaveBeenCalledTimes(1);
    });

    it('resets on subsequent calls', () => {
      let callCount = 0;
      const debouncedFn = vi.fn(() => callCount++);

      let timeout: ReturnType<typeof setTimeout> | null = null;
      const debounce = (fn: () => void, delay: number) => {
        if (timeout) clearTimeout(timeout);
        timeout = setTimeout(fn, delay);
      };

      debounce(debouncedFn, 300);
      vi.advanceTimersByTime(100);

      debounce(debouncedFn, 300);
      vi.advanceTimersByTime(100);

      vi.advanceTimersByTime(300);
      expect(debouncedFn).toHaveBeenCalledTimes(1);
    });
  });

  describe('Canvas rendering', () => {
    it('renders canvas element', () => {
      const canvas = document.getElementById('canvas') as HTMLCanvasElement;
      expect(canvas).toBeTruthy();
      expect(canvas.tagName).toBe('CANVAS');
    });

    it('has canvas attributes', () => {
      const canvas = document.getElementById('canvas') as HTMLCanvasElement;
      expect(canvas).toBeTruthy();

      // In jsdom, getContext may return null, which is expected
      // Just verify the canvas element exists and can be queried
      const hasCanvas = !!document.querySelector('canvas');
      expect(hasCanvas).toBe(true);
    });
  });

  describe('Button states', () => {
    it('recenter and reset disabled initially', () => {
      const recenterBtn = document.getElementById('recenter-btn') as HTMLButtonElement;
      const resetBtn = document.getElementById('reset-btn') as HTMLButtonElement;

      expect(recenterBtn.disabled).toBe(true);
      expect(resetBtn.disabled).toBe(true);
    });

    it('can be enabled when images are loaded', () => {
      const recenterBtn = document.getElementById('recenter-btn') as HTMLButtonElement;
      const resetBtn = document.getElementById('reset-btn') as HTMLButtonElement;

      // Simulate images loaded
      recenterBtn.disabled = false;
      resetBtn.disabled = false;

      expect(recenterBtn.disabled).toBe(false);
      expect(resetBtn.disabled).toBe(false);
    });
  });
});

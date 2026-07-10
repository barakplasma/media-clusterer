import { describe, it, expect } from 'vitest';
import { asFloat32 } from './db';

describe('asFloat32 (legacy cache entry conversion)', () => {
  it('passes Float32Array through unchanged', () => {
    const v = Float32Array.from([0.1, 0.2, 0.3]);
    expect(asFloat32(v)).toBe(v);
  });

  it('converts legacy Float64Array entries to Float32Array', () => {
    const legacy = Float64Array.from([0.5, -0.25, 1]);
    const out = asFloat32(legacy);
    expect(out).toBeInstanceOf(Float32Array);
    expect(Array.from(out!)).toEqual([0.5, -0.25, 1]);
  });

  it('returns null for missing or unexpected values', () => {
    expect(asFloat32(undefined)).toBeNull();
    expect(asFloat32(null)).toBeNull();
    expect(asFloat32('junk')).toBeNull();
  });
});


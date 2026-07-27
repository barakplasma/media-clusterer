/**
 * Performance benchmarks for the compute hot paths.
 *
 * Run with: npm run bench
 *
 * Sizes model real usage: 1k = typical folder, 5k = large folder,
 * 20k = stress case for "large folder support".
 */

import { bench, describe } from "vitest";
import {
  kmeansAsync,
  spreadPointsAsync,
  generateMetadataBasedLayout,
  cullAndPrioritize,
} from "./compute";
import type { Point } from "./types";

// Deterministic PRNG so runs are comparable
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makePoints2D(n: number, spread = 4000): number[][] {
  const rnd = mulberry32(42);
  return Array.from({ length: n }, () => [
    (rnd() - 0.5) * spread,
    (rnd() - 0.5) * spread,
  ]);
}

function makeVectors(n: number, dims = 768): Float32Array[] {
  const rnd = mulberry32(7);
  return Array.from({ length: n }, () => {
    const v = new Float32Array(dims);
    for (let i = 0; i < dims; i++) v[i] = rnd() - 0.5;
    return v;
  });
}

function makeFiles(n: number, folders = 40) {
  const rnd = mulberry32(11);
  return Array.from({ length: n }, (_, i) => ({
    name: `folder${Math.floor(rnd() * folders)}/img${i}.jpg`,
    lastModified: 1700000000000 + Math.floor(rnd() * 1e9),
  }));
}

const noYield = () => Promise.resolve();

for (const n of [1000, 5000, 20000]) {
  describe(`spreadPointsAsync n=${n}`, () => {
    const pts = makePoints2D(n);
    bench("spread", async () => {
      await spreadPointsAsync(pts, 1.0, noYield);
    });
  });

  describe(`kmeansAsync n=${n}`, () => {
    const pts = makePoints2D(n);
    const k = Math.max(2, Math.round(Math.sqrt(n / 2)));
    bench(`kmeans k=${k}`, async () => {
      await kmeansAsync(pts, k, 60, noYield);
    });
  });

  describe(`generateMetadataBasedLayout n=${n}`, () => {
    const files = makeFiles(n);
    bench("layout", () => {
      generateMetadataBasedLayout(files);
    });
  });

  describe(`cullAndPrioritize n=${n}`, () => {
    const pts = makePoints2D(n) as unknown as Point[];
    // Zoomed out: everything visible, forces the prioritization sort
    const camera = { x: 0, y: 0, scale: 0.2 };
    const rank = new Int32Array(n).fill(n);
    for (let i = 0; i < 100 && i < n; i++) rank[(i * 7) % n] = i;
    bench("cull+sort over budget (search active)", () => {
      cullAndPrioritize(pts, camera, 1920, 1080, 4.8, rank, 400);
    });
    bench("cull+sort over budget (no search)", () => {
      cullAndPrioritize(pts, camera, 1920, 1080, 4.8, null, 400);
    });
  });
}

describe("projection input conversion (n=5000, 768 dims)", () => {
  const vectors = makeVectors(5000);
  bench("Array.from copy (old runProjection path)", () => {
    vectors.map((v) => Array.from(v));
  });
  bench("Float64Array row copy (new path)", () => {
    vectors.map((v) => Float64Array.from(v));
  });
});

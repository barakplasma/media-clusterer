/**
 * Boot smoke test: import the real app module against the real index.html
 * DOM. Catches module-init regressions (missing DOM elements, top-level
 * throws) that the unit tests — which never import app.ts — can't see.
 */
import { readFileSync } from 'fs';
import { it, expect } from 'vitest';

it('app module boots against the real index.html DOM without throwing', async () => {
  const html = readFileSync('index.html', 'utf-8');
  const bodyMatch = html.match(/<body[^>]*>([\s\S]*)<\/body>/);
  expect(bodyMatch).toBeTruthy();
  document.body.innerHTML = bodyMatch![1].replace(/<script[\s\S]*?<\/script>/g, '');

  // jsdom doesn't implement <dialog> showModal/close
  (HTMLDialogElement.prototype as { showModal: () => void }).showModal ??= function () {};
  (HTMLDialogElement.prototype as { close: () => void }).close ??= function () {};

  await import('./app');

  const status = document.getElementById('status');
  expect(status).toBeTruthy();
});

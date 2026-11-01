// Solid-rendered app chrome: status line, progress bars, toast. The canvas, workers and
// the rest of app.ts stay imperative; they only call setStatus/setProgress/showToast.
import { createSignal } from 'solid-js'
import { render } from 'solid-js/web'

const [status, setStatusSig] = createSignal('Loading library…')
const [progress, setProgressSig] = createSignal(0)
const [toast, setToast] = createSignal({ message: '', level: 'info', visible: false })

export const setStatus = setStatusSig
export const setProgress = (pct: number) => setProgressSig(Math.min(100, Math.max(0, pct)))

let toastTimer: ReturnType<typeof setTimeout> | undefined
export function showToast(message: string, level: 'info' | 'warn' | 'error' = 'info', durationMs = 5000) {
  setToast({ message, level, visible: true })
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => setToast((t) => ({ ...t, visible: false })), durationMs)
}

const mount = (id: string, view: () => unknown) => {
  const el = document.getElementById(id)!
  el.textContent = ''
  render(view as () => never, el)
}

export function mountChrome() {
  mount('status', () => status())
  mount('welcome-status', () => status())
  mount('progress-wrap', () => <div id="progress-bar" style={{ width: `${progress()}%` }} />)
  mount('welcome-bar-wrap', () => <div id="welcome-bar" style={{ width: `${progress()}%` }} />)
  mount('toast-root', () => (
    <div id="toast" class={`${toast().visible ? 'visible' : ''} ${toast().level === 'info' ? '' : toast().level}`.trim()}>
      {toast().message}
    </div>
  ))
}

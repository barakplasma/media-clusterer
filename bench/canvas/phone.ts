// Phone-friendly runner: UI over window.bench from bench.ts. Open /canvas-bench.html on the device.
import './bench'

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const base = ['logic-baseline', 'logic-grid', 'baseline-main', 'opt2d-main', 'opt2d-worker', 'pixi-main']
const heavy = ['opt2d-all', 'pixi-all']
const rows: Record<string, any>[] = []

function gpu() {
  try {
    const gl = document.createElement('canvas').getContext('webgl2')!
    const ext = gl.getExtension('WEBGL_debug_renderer_info')
    return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'webgl2 (renderer hidden)'
  } catch { return 'no webgl2' }
}
const info = {
  ua: navigator.userAgent, cores: navigator.hardwareConcurrency, memGB: (navigator as any).deviceMemory ?? '?',
  dpr: devicePixelRatio, gpu: gpu(),
}
$('info').textContent = Object.entries(info).map(([k, v]) => `${k}: ${v}`).join('\n')

const cols = ['n', 'renderer', 'p50', 'p95', 'max', 'fps', 'drawn', 'mainLateP95']
const f = (v: unknown) => (typeof v === 'number' ? (v >= 100 ? v.toFixed(0) : v.toFixed(1)) : String(v))
function draw() {
  $('out').innerHTML = `<tr>${cols.map((c) => `<th>${c}</th>`).join('')}</tr>` +
    rows.map((r) => `<tr>${cols.map((c) => `<td>${f(r[c])}</td>`).join('')}</tr>`).join('')
}

$('run').onclick = async () => {
  const sizes = [...document.querySelectorAll<HTMLInputElement>('input[name=n]:checked')].map((e) => +e.value)
  const names = base.concat($<HTMLInputElement>('heavy').checked ? heavy : [])
  const flush = $<HTMLInputElement>('flush').checked
  ;($('run') as HTMLButtonElement).disabled = true
  for (const n of sizes) for (const name of names) {
    $('status').textContent = `running ${name} @ ${n}…`
    try {
      const r = await (window as any).bench(name, n, flush)
      rows.push({ n, renderer: name, p50: r.p50, p95: r.p95, max: r.max, fps: 1000 / r.p50, drawn: r.drawn, mainLateP95: r.mainLateP95 ?? 0 })
    } catch (e) { rows.push({ n, renderer: name, p50: NaN, p95: NaN, max: NaN, fps: NaN, drawn: 0, mainLateP95: 0, err: String(e) }) }
    draw()
  }
  $('status').textContent = `done (raster ${flush ? 'included' : 'excluded'})`
  ;($('run') as HTMLButtonElement).disabled = false
  ;($('copy') as HTMLButtonElement).disabled = false
}
$('copy').onclick = () => navigator.clipboard.writeText(JSON.stringify({ info, flush: $<HTMLInputElement>('flush').checked, rows }, null, 1))

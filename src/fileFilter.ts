/**
 * Pre-load filters for big folders: type, dates, file size, video length and
 * embedding-cache status. Cheap checks run first; EXIF dates and video
 * durations are read only when a filter needs them, and cached on the file.
 */

import exifr from '@modernized/exifr'
import pLimit from 'p-limit'
import type { PhotoFile } from './types'

export type EmbeddedFilter = 'any' | 'current' | 'some-model' | 'none'

export interface FileFilter {
  images: boolean
  videos: boolean
  modifiedFrom: string // 'YYYY-MM-DD' or '' (inclusive, local time)
  modifiedTo: string
  createdFrom: string
  createdTo: string
  minSizeMB: number | null
  maxSizeMB: number | null
  minDurationSec: number | null // videos only; images pass
  maxDurationSec: number | null
  embedded: EmbeddedFilter
}

export const DEFAULT_FILTER: FileFilter = {
  images: true,
  videos: true,
  modifiedFrom: '',
  modifiedTo: '',
  createdFrom: '',
  createdTo: '',
  minSizeMB: null,
  maxSizeMB: null,
  minDurationSec: null,
  maxDurationSec: null,
  embedded: 'any'
}

/** Keys (`name:size:lastModified`, no model prefix) already embedded. */
export interface EmbeddedSets {
  current: Set<string>
  anyModel: Set<string>
}

export const fileKey = (f: PhotoFile) => `${f.name}:${f.size}:${f.lastModified}`

export const isVideoName = (name: string) => /\.(mp4|webm)$/i.test(name)

/**
 * Split raw cache keys into the current model's set and the any-model set.
 * Prefixes are '' (nomic) or '@…/'; an OpenAI model name can itself hold '/',
 * so every suffix after a '/' counts for the any-model set.
 * shortcut: a root file and a subfolder file with the same name, size and mtime alias; fine for photo libraries.
 */
export function buildEmbeddedSets(keys: string[], currentPrefix: string): EmbeddedSets {
  const current = new Set<string>()
  const anyModel = new Set<string>()
  for (const k of keys) {
    if (currentPrefix ? k.startsWith(currentPrefix) : !k.startsWith('@')) current.add(k.slice(currentPrefix.length))
    if (!k.startsWith('@')) {
      anyModel.add(k)
      continue
    }
    for (let i = k.indexOf('/'); i !== -1; i = k.indexOf('/', i + 1)) anyModel.add(k.slice(i + 1))
  }
  return { current, anyModel }
}

const dayStart = (s: string) => (s ? new Date(`${s}T00:00:00`).getTime() : -Infinity)
const dayEnd = (s: string) => (s ? new Date(`${s}T23:59:59.999`).getTime() : Infinity)
const inRange = (v: number, lo: number, hi: number) => v >= lo && v <= hi

export const needsCreated = (f: FileFilter) => !!(f.createdFrom || f.createdTo)
export const needsDuration = (f: FileFilter) => f.minDurationSec != null || f.maxDurationSec != null

/** EXIF Date Taken, falling back to modified when absent (videos, screenshots). */
export async function readCreated(f: PhotoFile): Promise<number> {
  if (f.created !== undefined) return f.created
  let t = NaN
  if (!isVideoName(f.name)) {
    try {
      const exif = await exifr.parse(f.file, { pick: ['DateTimeOriginal', 'CreateDate'] })
      const d = exif?.DateTimeOriginal ?? exif?.CreateDate
      if (d instanceof Date) t = d.getTime()
    } catch {
      // unreadable EXIF falls back below
    }
  }
  f.created = Number.isFinite(t) ? t : f.lastModified
  return f.created
}

/** Video duration in seconds; NaN when the browser can't tell. */
export async function readDuration(f: PhotoFile): Promise<number> {
  if (f.duration !== undefined) return f.duration
  const url = URL.createObjectURL(f.file)
  const video = document.createElement('video')
  video.preload = 'metadata'
  video.muted = true
  try {
    f.duration = await new Promise<number>((resolve) => {
      const timer = setTimeout(() => resolve(NaN), 8000)
      const finish = (v: number) => {
        clearTimeout(timer)
        resolve(v)
      }
      video.onerror = () => finish(NaN)
      video.onloadedmetadata = () => {
        if (Number.isFinite(video.duration)) return finish(video.duration)
        // MediaRecorder webm has no duration header; seeking to the end makes Chrome compute it.
        video.ondurationchange = () => Number.isFinite(video.duration) && finish(video.duration)
        video.currentTime = 1e9
      }
      video.src = url
    })
  } finally {
    video.removeAttribute('src')
    video.load()
    URL.revokeObjectURL(url)
  }
  return f.duration
}

const exifLimit = pLimit(8)
// Chrome caps concurrent media players, same reason as videoFrameLimit in app.ts.
const videoLimit = pLimit(4)

/**
 * Files passing `filter`. Returns null when `isStale()` turns true midway, so
 * a newer call can replace this one.
 */
export async function applyFileFilter(
  files: PhotoFile[],
  filter: FileFilter,
  embedded: EmbeddedSets,
  onProgress?: (done: number, total: number) => void,
  isStale: () => boolean = () => false
): Promise<PhotoFile[] | null> {
  const modLo = dayStart(filter.modifiedFrom)
  const modHi = dayEnd(filter.modifiedTo)
  const sizeLo = filter.minSizeMB != null ? filter.minSizeMB * 1024 * 1024 : -Infinity
  const sizeHi = filter.maxSizeMB != null ? filter.maxSizeMB * 1024 * 1024 : Infinity

  const cheap = files.filter((f) => {
    const video = isVideoName(f.name)
    if (video ? !filter.videos : !filter.images) return false
    if (!inRange(f.lastModified, modLo, modHi)) return false
    if (!inRange(f.size, sizeLo, sizeHi)) return false
    if (filter.embedded === 'any') return true
    const k = fileKey(f)
    if (filter.embedded === 'current') return embedded.current.has(k)
    if (filter.embedded === 'some-model') return embedded.anyModel.has(k)
    return !embedded.anyModel.has(k)
  })

  const wantCreated = needsCreated(filter)
  const wantDuration = needsDuration(filter)
  if (!wantCreated && !wantDuration) return cheap

  const crLo = dayStart(filter.createdFrom)
  const crHi = dayEnd(filter.createdTo)
  const durLo = filter.minDurationSec ?? -Infinity
  const durHi = filter.maxDurationSec ?? Infinity
  let done = 0
  const keep = await Promise.all(
    cheap.map(async (f) => {
      // Checked inside the limiters so a stale run drains its queue without reading files.
      let ok = true
      if (wantCreated) ok = inRange(await exifLimit(() => (isStale() ? NaN : readCreated(f))), crLo, crHi)
      // Unknown durations fail a length filter rather than sneak through it.
      if (ok && wantDuration && isVideoName(f.name))
        ok = inRange(await videoLimit(() => (isStale() ? NaN : readDuration(f))), durLo, durHi)
      onProgress?.(++done, cheap.length)
      return ok
    })
  )
  if (isStale()) return null
  return cheap.filter((_, i) => keep[i])
}

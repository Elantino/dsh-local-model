/**
 * Engine provisioning: fetch an official llama.cpp build into the DSH data
 * directory and point the spawner at it.
 *
 * Why download instead of bundling: the release is platform- and GPU-specific
 * (CPU / Vulkan / CUDA, x64 / arm64) and a CUDA build drags in hundreds of
 * megabytes of runtime DLLs. Pinning one variant inside the npm package would
 * be wrong for almost every user, and the npm package must stay small enough to
 * install quickly. So the plugin downloads the matching asset once, verifies
 * the SHA-256 that GitHub publishes for it, and keeps it under
 * `$DSH_HOME/local-model/engine/`.
 *
 * The default is the Vulkan build: one self-contained ~32 MB zip that uses the
 * GPU on NVIDIA/AMD/Intel, without the CUDA runtime download. CPU is the
 * fallback, CUDA is opt-in for people who already have the runtime.
 *
 * @module dsh-local-model/engine
 */

import { createHash } from 'node:crypto'
import { existsSync } from 'node:fs'
import { mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'

import { gunzipSync, unzipSync } from 'fflate'

/** GitHub's API root for the project this plugin drives. */
const RELEASES_API = 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15'

/** How long release metadata is reused before another API call. */
const METADATA_TTL_MS = 60 * 60 * 1000

/** @type {{ at: number, release: Record<string, any> | undefined }} */
let metadataCache = { at: 0, release: undefined }

/** Drop cached release metadata (tests and the remove flow). */
export function resetEngineMetadata() {
  metadataCache = { at: 0, release: undefined }
}

/** Live provisioning progress, read by the settings API. */
const state = {
  phase: 'idle',
  received: 0,
  total: 0,
  tag: '',
  asset: '',
  error: '',
  startedAt: 0,
}

/**
 * The harness home used for plugin data.
 * @param {NodeJS.ProcessEnv} [env] - environment to read.
 * @returns {string} absolute DSH home.
 */
export function dshHome(env = process.env) {
  return env.DSH_HOME && env.DSH_HOME.trim() !== ''
    ? env.DSH_HOME
    : join(homedir(), '.dsh')
}

/**
 * Directory that holds downloaded engines.
 * @param {NodeJS.ProcessEnv} [env] - environment to read.
 * @returns {string} absolute engine root.
 */
export function engineRoot(env = process.env) {
  return join(dshHome(env), 'local-model', 'engine')
}

/**
 * Asset-name patterns for one platform, in preference order.
 * @param {string} platform - `process.platform` value.
 * @param {string} arch - `process.arch` value.
 * @param {'auto' | 'vulkan' | 'cpu' | 'cuda'} [variant] - requested variant.
 * @returns {RegExp[]} patterns matched against release asset names.
 */
export function assetPatterns(platform, arch, variant = 'auto') {
  const bits = arch === 'arm64' ? 'arm64' : 'x64'
  const wanted = variant === 'auto' ? ['vulkan', 'cpu'] : [variant]
  /** @type {RegExp[]} */
  const patterns = []
  for (const kind of wanted) {
    if (platform === 'win32') {
      patterns.push(new RegExp(`-bin-win-${kind === 'cuda' ? 'cuda-13\\.4' : kind}-${bits}\\.zip$`))
      if (kind === 'cuda') patterns.push(new RegExp(`-bin-win-cuda-12\\.4-${bits}\\.zip$`))
    } else if (platform === 'darwin') {
      // macOS builds have Metal compiled in; there is no variant suffix.
      patterns.push(new RegExp(`-bin-macos-${bits}\\.tar\\.gz$`))
    } else if (kind === 'cuda') {
      patterns.push(new RegExp(`-bin-ubuntu-cuda-13\\.4-${bits}\\.tar\\.gz$`))
      patterns.push(new RegExp(`-bin-ubuntu-cuda-12\\.8-${bits}\\.tar\\.gz$`))
    } else if (kind === 'vulkan') {
      patterns.push(new RegExp(`-bin-ubuntu-vulkan-${bits}\\.tar\\.gz$`))
    } else {
      // The Linux CPU asset carries no `cpu` infix: `…-bin-ubuntu-x64.tar.gz`.
      patterns.push(new RegExp(`-bin-ubuntu-${bits}\\.tar\\.gz$`))
    }
  }
  return patterns
}

/**
 * Pick the first release that carries an asset matching this machine.
 * @param {{ fetchImpl?: typeof fetch, platform?: string, arch?: string, variant?: string, now?: number }} [options] - overrides.
 * @returns {Promise<{ tag: string, asset: Record<string, any>, patterns: RegExp[] }>} the choice.
 * @throws {Error} when no release offers a usable binary.
 */
export async function resolveRelease(options = {}) {
  const fetchImpl = options.fetchImpl ?? fetch
  const platform = options.platform ?? process.platform
  const arch = options.arch ?? process.arch
  const variant = options.variant ?? 'auto'
  const now = options.now ?? Date.now()
  if (metadataCache.release === undefined || now - metadataCache.at > METADATA_TTL_MS) {
    const response = await fetchImpl(RELEASES_API, {
      headers: {
        accept: 'application/vnd.github+json',
        'user-agent': 'dsh-local-model',
        'x-github-api-version': '2022-11-28',
      },
      signal: AbortSignal.timeout(30_000),
    })
    if (!response.ok) throw new Error(`GitHub releases API 返回 HTTP ${response.status}`)
    const releases = await response.json()
    if (!Array.isArray(releases)) throw new Error('GitHub releases API 返回了意外的数据')
    metadataCache = { at: now, release: { releases } }
  }
  const patterns = assetPatterns(platform, arch, variant)
  for (const release of metadataCache.release.releases) {
    if (release?.draft === true) continue
    const assets = Array.isArray(release.assets) ? release.assets : []
    for (const pattern of patterns) {
      const asset = assets.find((entry) => typeof entry?.name === 'string' && pattern.test(entry.name))
      if (asset !== undefined) return { tag: String(release.tag_name), asset, patterns }
    }
  }
  throw new Error(`没有找到适用于 ${platform}/${arch}（${variant}）的 llama.cpp 预编译包；可改用 backend: openai 连接已有服务`)
}

/**
 * Current provisioning status for the settings API.
 * @returns {Record<string, any>} a detached copy.
 */
export function engineStatus() {
  return { ...state }
}

/**
 * Read the installed engine manifest, if any.
 * @param {NodeJS.ProcessEnv} [env] - environment to read.
 * @returns {Promise<Record<string, any> | undefined>} `{ binary, tag, asset, variant, dir }`.
 */
export async function findEngine(env = process.env) {
  const root = engineRoot(env)
  let entries
  try {
    entries = await readdir(root, { withFileTypes: true })
  } catch {
    return undefined
  }
  for (const entry of entries) {
    if (!entry.isDirectory()) continue
    const dir = join(root, entry.name)
    try {
      const manifest = JSON.parse(await readFile(join(dir, 'engine.json'), 'utf8'))
      const binary = join(dir, manifest.binary)
      if (typeof manifest.binary === 'string' && existsSync(binary)) {
        return { ...manifest, dir, binary, tag: manifest.tag ?? entry.name }
      }
    } catch {
      // A half-written or stale directory is simply not an engine.
    }
  }
  return undefined
}

/**
 * Download and unpack the matching engine.
 * @param {{
 *   variant?: 'auto' | 'vulkan' | 'cpu' | 'cuda',
 *   fetchImpl?: typeof fetch,
 *   env?: NodeJS.ProcessEnv,
 *   platform?: string,
 *   arch?: string,
 *   onProgress?: (state: Record<string, any>) => void,
 *   signal?: AbortSignal,
 * }} [options] - provisioning inputs.
 * @returns {Promise<Record<string, any>>} the installed manifest.
 */
export async function provisionEngine(options = {}) {
  const fetchImpl = options.fetchImpl ?? fetch
  const env = options.env ?? process.env
  const report = (patch) => {
    Object.assign(state, patch)
    options.onProgress?.({ ...state })
  }

  report({
    phase: 'metadata', error: '', received: 0, total: 0, tag: '', asset: '', startedAt: Date.now(),
  })
  try {
    const { tag, asset, patterns } = await resolveRelease({
      fetchImpl,
      ...(options.platform === undefined ? {} : { platform: options.platform }),
      ...(options.arch === undefined ? {} : { arch: options.arch }),
      ...(options.variant === undefined ? {} : { variant: options.variant }),
    })
    const dir = join(engineRoot(env), tag)
    const existing = await findEngine(env)
    if (existing?.tag === tag) {
      report({ phase: 'done', tag, asset: existing.asset ?? '', total: 1, received: 1 })
      return existing
    }

    report({ phase: 'download', tag, asset: asset.name, total: Number(asset.size) || 0 })
    const buffer = await download(asset, fetchImpl, (received, total) => {
      report({ received, total })
    }, options.signal)

    if (typeof asset.digest === 'string' && asset.digest.startsWith('sha256:')) {
      const actual = createHash('sha256').update(buffer).digest('hex')
      if (actual !== asset.digest.slice('sha256:'.length)) {
        throw new Error(`下载校验失败：${asset.name} 的 SHA-256 与 GitHub 记录不一致`)
      }
    }

    report({ phase: 'extract' })
    await mkdir(dir, { recursive: true })
    const files = asset.name.endsWith('.zip') ? unzipSync(buffer) : extractTarGz(buffer)
    for (const [name, bytes] of Object.entries(files)) {
      const relative = safeRelative(name)
      if (relative === undefined) continue
      const target = join(dir, relative)
      await mkdir(dirname(target), { recursive: true })
      await writeFile(target, bytes)
    }

    const binary = await locateBinary(dir)
    if (binary === undefined) throw new Error('下载的压缩包里没有 llama-server')
    const kind = /-(vulkan|cuda|rocm|sycl|openvino)-/.exec(asset.name)?.[1]
    const manifest = {
      tag,
      asset: asset.name,
      variant: kind ?? (String(options.platform ?? process.platform) === 'darwin' ? 'metal' : 'cpu'),
      binary: binary.slice(dir.length + 1),
      installedAt: new Date().toISOString(),
    }
    await writeFile(join(dir, 'engine.json'), JSON.stringify(manifest, null, 2))
    report({ phase: 'done', received: 1, total: 1 })
    return { ...manifest, dir, binary }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    report({ phase: 'error', error: message })
    throw error
  }
}

/**
 * Remove every downloaded engine.
 * @param {NodeJS.ProcessEnv} [env] - environment to read.
 * @returns {Promise<void>} settlement.
 */
export async function removeEngines(env = process.env) {
  await rm(engineRoot(env), { recursive: true, force: true })
  metadataCache = { at: 0, release: undefined }
  Object.assign(state, { phase: 'idle', received: 0, total: 0, tag: '', asset: '', error: '' })
}

/**
 * Stream one asset into memory with progress and a hard size bound.
 * @param {Record<string, any>} asset - release asset.
 * @param {typeof fetch} fetchImpl - transport.
 * @param {(received: number, total: number) => void} onProgress - byte progress.
 * @param {AbortSignal | undefined} signal - caller cancellation.
 * @returns {Promise<Buffer>} the downloaded bytes.
 */
async function download(asset, fetchImpl, onProgress, signal) {
  const response = await fetchImpl(asset.browser_download_url, {
    headers: { 'user-agent': 'dsh-local-model' },
    redirect: 'follow',
    signal: signal === undefined ? AbortSignal.timeout(30 * 60_000) : AbortSignal.any([signal, AbortSignal.timeout(30 * 60_000)]),
  })
  if (!response.ok) throw new Error(`下载 ${asset.name} 失败：HTTP ${response.status}`)
  const total = Number(asset.size) || Number(response.headers.get('content-length')) || 0
  if (total > 800 * 1024 * 1024) throw new Error(`引擎包过大（${Math.round(total / 1024 / 1024)} MB），已拒绝下载`)
  const chunks = []
  let received = 0
  for await (const chunk of response.body) {
    received += chunk.length
    chunks.push(Buffer.from(chunk))
    onProgress(received, total)
  }
  return Buffer.concat(chunks)
}

/**
 * Reject entries that would escape the target directory.
 * @param {string} name - archive entry name.
 * @returns {string | undefined} a safe relative path, or undefined to skip it.
 */
function safeRelative(name) {
  const normalized = String(name).replace(/\\/g, '/')
  if (normalized === '' || normalized.endsWith('/')) return undefined
  const parts = normalized.split('/').filter((part) => part !== '' && part !== '.')
  if (parts.length === 0) return undefined
  if (parts.some((part) => part === '..' || /^[A-Za-z]:$/.test(part))) return undefined
  return parts.join('/')
}

/**
 * Find the server binary anywhere under the unpacked directory.
 * @param {string} dir - unpack root.
 * @param {number} [depth] - remaining recursion depth.
 * @returns {Promise<string | undefined>} absolute binary path.
 */
async function locateBinary(dir, depth = 4) {
  const wanted = process.platform === 'win32' ? 'llama-server.exe' : 'llama-server'
  const entries = await readdir(dir, { withFileTypes: true })
  for (const entry of entries) {
    if (entry.isFile() && entry.name === wanted) return join(dir, entry.name)
  }
  if (depth === 0) return undefined
  for (const entry of entries) {
    if (!entry.isDirectory()) continue
    const found = await locateBinary(join(dir, entry.name), depth - 1)
    if (found !== undefined) return found
  }
  return undefined
}

/**
 * Read a fixed-width ASCII field from a tar header.
 * @param {Uint8Array} header - 512-byte header.
 * @param {number} offset - field start.
 * @param {number} length - field width.
 * @returns {string} the trimmed text.
 */
function tarField(header, offset, length) {
  let end = offset + length
  while (end > offset && header[end - 1] === 0) end -= 1
  return Buffer.from(header.subarray(offset, end)).toString('utf8').trim()
}

/**
 * Minimal ustar reader for the `.tar.gz` releases.
 * @param {Buffer} buffer - gzip bytes.
 * @returns {Record<string, Uint8Array>} regular files by relative name.
 */
function extractTarGz(buffer) {
  const tar = gunzipSync(buffer)
  /** @type {Record<string, Uint8Array>} */
  const files = {}
  let offset = 0
  while (offset + 512 <= tar.length) {
    const header = tar.subarray(offset, offset + 512)
    if (header.every((byte) => byte === 0)) break
    const name = tarField(header, 0, 100)
    const prefix = tarField(header, 345, 155)
    const full = prefix === '' ? name : `${prefix}/${name}`
    const size = Number.parseInt(tarField(header, 124, 12) || '0', 8)
    const type = header[156]
    const dataStart = offset + 512
    if ((type === 0 || type === 48 || type === undefined) && Number.isFinite(size) && size > 0) {
      files[full] = tar.subarray(dataStart, dataStart + size)
    }
    offset = dataStart + Math.ceil(size / 512) * 512
  }
  return files
}

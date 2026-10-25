/**
 * llama.cpp process manager.
 *
 * One `llama-server` per model, started on first use, reused by every later
 * request, and stopped again after an idle period. That trades a cold-start
 * (the model has to be loaded from disk) for predictable memory: a machine
 * only pays for the models it actually used recently.
 *
 * Ports are claimed from a small private range above `basePort`, so two
 * models can be resident at once and nothing depends on a hard-coded port.
 * Startup waits for `/health`, which is what llama.cpp answers once the model
 * is loaded and the HTTP API is accepting work.
 *
 * Security posture: the child binds to `127.0.0.1` only, receives no shell
 * string (arguments are passed as an array), and its stdout/stderr are
 * captured into a bounded ring for diagnostics instead of being inherited.
 *
 * @module dsh-local-model/llama
 */

import { spawn as nodeSpawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { createServer } from 'node:net'
import { delimiter, join } from 'node:path'

import { LlmError } from '@deepseek-ai/dsh-llm'
import { LOCAL_RUNTIME_MISSING, abortedFailure, missingRuntimeFailure, runtimeFailure } from './errors.js'
import { findEngine } from './engine.js'
import { mmprojWarning } from './gguf.js'

/** How long a stopped server has to exit before it is killed. */
const DEFAULT_STOP_GRACE_MS = 5_000

/** Gap between health probes while a server loads. */
const HEALTH_POLL_MS = 250

/** Bytes of process output retained for diagnostics. */
const STDERR_TAIL_BYTES = 8_192

/**
 * Well-known places `llama-server` is unpacked when it is not on `PATH`.
 *
 * A llama.cpp release is a zip that people unpack into a folder of their own,
 * so "not on PATH" is the common case rather than the exception. Probing a
 * short, predictable list keeps a GGUF-only setup working without asking the
 * user to edit PATH or paste a path.
 * @param {NodeJS.ProcessEnv} env - environment to derive the list from.
 * @param {string} [platform] - platform override (tests).
 * @returns {string[]} candidate paths, most likely first.
 */
export function commonLlamaServerPaths(env = process.env, platform = process.platform) {
  if (platform === 'win32') {
    const local = env.LOCALAPPDATA ?? ''
    const roaming = env.APPDATA ?? ''
    const home = env.USERPROFILE ?? ''
    return [
      'C:\\llama\\llama-server.exe',
      'D:\\llama\\llama-server.exe',
      'C:\\llama.cpp\\llama-server.exe',
      'D:\\llama.cpp\\llama-server.exe',
      ...(local === '' ? [] : [join(local, 'Microsoft', 'WinGet', 'Links', 'llama-server.exe')]),
      ...(roaming === '' ? [] : [join(roaming, 'npm', 'llama-server.exe')]),
      ...(home === '' ? [] : [join(home, 'scoop', 'shims', 'llama-server.exe')]),
    ]
  }
  const home = env.HOME ?? ''
  return [
    '/usr/local/bin/llama-server',
    '/opt/homebrew/bin/llama-server',
    '/usr/bin/llama-server',
    ...(home === '' ? [] : [join(home, '.local', 'bin', 'llama-server')]),
  ]
}

/**
 * Resolve the `llama-server` executable.
 * @param {string | undefined} configured - explicit path from config.
 * @param {NodeJS.ProcessEnv} [env] - environment to search.
 * @param {{ searchPaths?: string[] }} [options] - override the well-known list (tests).
 * @returns {string | undefined} the path to use, or undefined when none was found.
 */
export function resolveLlamaServer(configured, env = process.env, options = {}) {
  const explicit = typeof configured === 'string' ? configured.trim() : ''
  if (explicit !== '') return explicit
  const fromEnv = typeof env.LLAMA_SERVER === 'string' ? env.LLAMA_SERVER.trim() : ''
  if (fromEnv !== '') return fromEnv
  const name = process.platform === 'win32' ? 'llama-server.exe' : 'llama-server'
  const pathValue = typeof env.PATH === 'string' ? env.PATH : ''
  for (const directory of pathValue.split(delimiter)) {
    if (directory === '') continue
    const candidate = join(directory, name)
    if (existsSync(candidate)) return candidate
  }
  for (const candidate of options.searchPaths ?? commonLlamaServerPaths(env)) {
    if (candidate !== '' && existsSync(candidate)) return candidate
  }
  return undefined
}

/**
 * Build the argument vector for one spawned llama-server.
 * @param {Record<string, any>} model - the normalized model def.
 * @param {number} port - the port this instance owns.
 * @param {number} ctxSize - context size to allocate.
 * @returns {string[]} arguments, without the executable.
 */
export function buildLlamaArgs(model, port, ctxSize) {
  const args = [
    '--model', model.file,
    '--host', '127.0.0.1',
    '--port', String(port),
    '--ctx-size', String(ctxSize * (model.parallel ?? 1)),
  ]
  if (typeof model.gpuLayers === 'number') args.push('--n-gpu-layers', String(model.gpuLayers))
  if (typeof model.threads === 'number') args.push('--threads', String(model.threads))
  if (typeof model.batchSize === 'number') args.push('--batch-size', String(model.batchSize))
  args.push('--parallel', String(model.parallel ?? 1))
  if (model.jinja === true) args.push('--jinja')
  const cacheType = model.kvCacheType === 'q8_0' || model.kvCacheType === 'q4_0' ? model.kvCacheType : undefined
  if (cacheType !== undefined) args.push('-ctk', cacheType, '-ctv', cacheType)
  // Quantized KV needs Flash Attention in current llama.cpp builds; an explicit
  // `off` from the user still wins so a broken GPU can be worked around.
  if (model.flashAttention === 'on' || model.flashAttention === 'off') {
    args.push('-fa', model.flashAttention)
  } else if (cacheType !== undefined) {
    args.push('-fa', 'on')
  }
  if (typeof model.chatTemplate === 'string' && model.chatTemplate !== '') {
    args.push('--chat-template-file', model.chatTemplate)
  }
  args.push(...(Array.isArray(model.extraArgs) ? model.extraArgs : []))
  return args
}

/**
 * Test whether a TCP port can be bound on loopback.
 * @param {number} port - candidate port.
 * @param {string} [host] - interface to test.
 * @returns {Promise<boolean>} true when the port is free.
 */
export function isPortFree(port, host = '127.0.0.1') {
  return new Promise((resolve) => {
    const server = createServer()
    server.unref()
    server.once('error', () => resolve(false))
    server.listen({ port, host, exclusive: true }, () => {
      server.close(() => resolve(true))
    })
  })
}

/**
 * Sleep, without keeping the event loop alive.
 * @param {number} ms - delay.
 * @returns {Promise<void>} resolves after the delay.
 */
function sleep(ms) {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms)
    timer.unref?.()
  })
}

/**
 * Await a promise but reject as soon as the caller aborts.
 * @template T
 * @param {Promise<T>} promise - the work being awaited.
 * @param {AbortSignal | undefined} signal - caller cancellation.
 * @returns {Promise<T>} the original result, or an `ABORTED` failure.
 */
function raceAbort(promise, signal) {
  if (signal === undefined) return promise
  if (signal.aborted) return Promise.reject(abortedFailure('local-model startup'))
  return new Promise((resolve, reject) => {
    const onAbort = () => reject(abortedFailure('local-model startup'))
    signal.addEventListener('abort', onAbort, { once: true })
    promise.then(
      (value) => {
        signal.removeEventListener('abort', onAbort)
        resolve(value)
      },
      (error) => {
        signal.removeEventListener('abort', onAbort)
        reject(error)
      },
    )
  })
}

/**
 * Owns every spawned `llama-server` process of one provider route.
 */
export class LlamaServers {
  /** @type {Map<string, Record<string, any>>} */
  #entries = new Map()
  /** @type {Set<number>} */
  #ports = new Set()
  #disposed = false
  #allocation = Promise.resolve()

  /**
   * @param {{
   *   spawn?: typeof nodeSpawn,
   *   fetch?: typeof fetch,
   *   logger?: any,
   *   basePort?: number,
   *   startupTimeoutMs?: number,
   *   idleTimeoutMs?: number,
   *   stopGraceMs?: number,
   *   env?: NodeJS.ProcessEnv,
   *   exists?: (file: string) => boolean,
   *   searchPaths?: string[],
   * }} [options] - injectable runtime (tests replace `spawn`, `fetch` and the
   *   well-known binary locations).
   */
  constructor(options = {}) {
    this.spawn = options.spawn ?? nodeSpawn
    this.fetch = options.fetch ?? globalThis.fetch
    this.logger = options.logger
    this.basePort = options.basePort ?? 8177
    this.startupTimeoutMs = options.startupTimeoutMs ?? 180_000
    this.idleTimeoutMs = options.idleTimeoutMs ?? 600_000
    this.stopGraceMs = options.stopGraceMs ?? DEFAULT_STOP_GRACE_MS
    this.searchPaths = options.searchPaths
    this.env = options.env ?? process.env
    this.exists = options.exists ?? existsSync
  }

  /** @returns {string[]} ids of the models with a live server process. */
  get running() {
    return [...this.#entries.keys()]
  }

  /**
   * Ensure one model has a running server and lease it for one request.
   * @param {Record<string, any>} model - the normalized model def.
   * @param {number} ctxSize - context size the server should allocate.
   * @param {{ signal?: AbortSignal }} [options] - caller cancellation.
   * @returns {Promise<{ baseURL: string, release: () => void }>} endpoint and release handle.
   */
  async acquire(model, ctxSize, options = {}) {
    if (this.#disposed) throw new LlmError('local-model: provider is shutting down', 'ABORTED')
    let entry = this.#entries.get(model.id)
    if (entry === undefined) {
      entry = {
        modelId: model.id,
        model,
        port: undefined,
        child: undefined,
        ready: undefined,
        stderr: '',
        exit: undefined,
        spawnError: undefined,
        active: 0,
        timer: undefined,
      }
      this.#entries.set(model.id, entry)
      entry.ready = this.#start(entry, ctxSize).catch((error) => {
        this.#forget(entry)
        throw error
      })
      // Nobody may be awaiting yet (a concurrent acquire can be); keep the
      // rejection from becoming an unhandled rejection, then rethrow on await.
      entry.ready.catch(() => {})
    }
    await raceAbort(entry.ready, options.signal).catch((error) => {
      // The caller gave up, but the start continues. Give the finished server
      // the same idle treatment a completed lease would, so an abandoned cold
      // start cannot pin a model in memory until the plugin unloads.
      entry.ready.then(
        () => {
          if (entry.active === 0) this.#scheduleIdle(entry)
        },
        () => {},
      )
      throw error
    })
    entry.active += 1
    this.#clearIdle(entry)
    const baseURL = `http://127.0.0.1:${entry.port}/v1`
    let released = false
    return {
      baseURL,
      release: () => {
        if (released) return
        released = true
        entry.active = Math.max(0, entry.active - 1)
        if (entry.active === 0) this.#scheduleIdle(entry)
      },
    }
  }

  /** Stop every spawned server and release every port. */
  async dispose() {
    this.#disposed = true
    const entries = [...this.#entries.values()]
    await Promise.allSettled(entries.map((entry) => this.#stop(entry)))
    this.#entries.clear()
    this.#ports.clear()
  }

  /**
   * Start one server and wait until it reports healthy.
   * @param {Record<string, any>} entry - the entry being started.
   * @param {number} ctxSize - context size to allocate.
   */
  async #start(entry, ctxSize) {
    const model = entry.model
    const projection = mmprojWarning(model.file)
    if (projection !== undefined) throw new LlmError(`local-model: ${projection}`, 'INVALID_REQUEST')
    if (!this.exists(model.file)) {
      throw runtimeFailure(`model file not found: ${model.file}`, '')
    }
    const configured = resolveLlamaServer(model.llamaServer, this.env, this.searchPaths === undefined ? {} : { searchPaths: this.searchPaths })
    // Last resort: an engine this plugin downloaded earlier. It is checked
    // after every user-controlled location so an explicit path always wins.
    const binary = configured ?? (await findEngine(this.env))?.binary
    if (binary === undefined) throw missingRuntimeFailure(model.llamaServer === '' ? 'llama-server' : model.llamaServer)

    const port = await this.#allocatePort()
    if (this.#disposed) {
      this.#ports.delete(port)
      throw new LlmError('local-model: provider is shutting down', 'ABORTED')
    }
    entry.port = port
    const args = buildLlamaArgs(model, port, ctxSize)

    let child
    try {
      child = this.spawn(binary, args, { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true })
    } catch (error) {
      throw missingRuntimeFailure(binary, error)
    }
    entry.child = child
    const collect = (chunk) => {
      entry.stderr = (entry.stderr + String(chunk)).slice(-STDERR_TAIL_BYTES)
    }
    child.stdout?.on('data', collect)
    child.stderr?.on('data', collect)
    child.on('error', (error) => {
      entry.spawnError = error
    })
    child.on('exit', (code, signal) => {
      entry.exit = { code, signal }
      this.#clearIdle(entry)
      if (this.#entries.get(model.id) === entry) this.#entries.delete(model.id)
      this.#ports.delete(port)
    })

    await this.#waitReady(entry, binary)
    if (this.#disposed) {
      // The plugin unloaded while the model was loading; do not keep the
      // process alive behind a disposed owner.
      this.#forget(entry)
      throw new LlmError('local-model: provider is shutting down', 'ABORTED')
    }
    this.logger?.info?.(
      `local-model: started llama-server for "${model.id}" on 127.0.0.1:${port} (pid ${child.pid ?? '?'})`,
    )
  }

  /**
   * Poll `/health` until the server answers or the startup budget runs out.
   * @param {Record<string, any>} entry - the starting entry.
   * @param {string} binary - executable path, for diagnostics.
   */
  async #waitReady(entry, binary) {
    const deadline = Date.now() + this.startupTimeoutMs
    while (true) {
      if (entry.spawnError !== undefined) {
        const error = entry.spawnError
        throw new LlmError(
          `local-model: could not start "${binary}" (${error.code ?? error.message})`,
          LOCAL_RUNTIME_MISSING,
          { cause: error },
        )
      }
      if (entry.exit !== undefined) {
        throw runtimeFailure(
          `llama-server exited with code ${entry.exit.code ?? '?'} before becoming ready`,
          entry.stderr,
        )
      }
      try {
        const response = await this.fetch(`http://127.0.0.1:${entry.port}/health`, {
          signal: AbortSignal.timeout(2_000),
        })
        if (response.ok) return
      } catch {
        // Connection refused while loading is the normal path, not a failure.
      }
      if (Date.now() > deadline) {
        throw runtimeFailure(
          `llama-server did not become ready within ${Math.round(this.startupTimeoutMs / 1_000)}s`,
          entry.stderr,
        )
      }
      await sleep(HEALTH_POLL_MS)
    }
  }

  /**
   * Claim the next free port in the configured range.
   * @returns {Promise<number>} the claimed port.
   */
  async #allocatePort() {
    const previous = this.#allocation
    let unlock
    this.#allocation = new Promise((resolve) => { unlock = resolve })
    await previous
    try {
    for (let offset = 0; offset < 64; offset += 1) {
      const port = this.basePort + offset
      if (port > 65_535) break
      if (this.#ports.has(port)) continue
      if (await isPortFree(port)) {
        this.#ports.add(port)
        return port
      }
    }
    throw new LlmError(
      `local-model: no free loopback port in ${this.basePort}..${Math.min(65_535, this.basePort + 63)}`,
      'LOCAL_RUNTIME_FAILED',
    )
    } finally {
      unlock()
    }
  }

  /**
   * Arm the idle stop timer for one entry.
   * @param {Record<string, any>} entry - an entry with no active lease.
   */
  #scheduleIdle(entry) {
    if (this.idleTimeoutMs <= 0 || this.#disposed) return
    this.#clearIdle(entry)
    entry.timer = setTimeout(() => {
      entry.timer = undefined
      if (entry.active === 0) void this.#stop(entry)
    }, this.idleTimeoutMs)
    entry.timer.unref?.()
  }

  /**
   * Disarm the idle stop timer.
   * @param {Record<string, any>} entry - the entry to touch.
   */
  #clearIdle(entry) {
    if (entry.timer === undefined) return
    clearTimeout(entry.timer)
    entry.timer = undefined
  }

  /**
   * Stop one server and release its port.
   * @param {Record<string, any>} entry - the entry to stop.
   */
  async #stop(entry) {
    this.#clearIdle(entry)
    if (this.#entries.get(entry.modelId) === entry) this.#entries.delete(entry.modelId)
    const child = entry.child
    if (child === undefined || child.exitCode !== null) {
      if (entry.port !== undefined) this.#ports.delete(entry.port)
      return
    }
    const exited = new Promise((resolve) => child.once('exit', resolve))
    try {
      child.kill()
    } catch {
      return
    }
    await Promise.race([exited, sleep(this.stopGraceMs)])
    if (child.exitCode === null) {
      try {
        child.kill('SIGKILL')
      } catch {
        // Already gone; nothing left to release.
      }
    }
    this.logger?.info?.(`local-model: stopped llama-server for "${entry.modelId}"`)
  }

  /**
   * Drop one entry without waiting for a graceful stop (startup failure path).
   * @param {Record<string, any>} entry - the failed entry.
   */
  #forget(entry) {
    // Keep the port reserved until exit, and escalate a stalled termination.
    void this.#stop(entry).catch(() => {})
  }
}

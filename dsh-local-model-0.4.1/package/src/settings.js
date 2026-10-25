import fs from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { randomUUID } from 'node:crypto'
import { Config, normalizeModels } from './config.js'
import { mmprojWarning, readGgufMetadata } from './gguf.js'
import { detectOllama, inspectOllama, ollamaRoot } from './ollama.js'
import { chatCompletionsUrl } from './openai.js'
import { resolveLlamaServer } from './llama.js'
import { engineStatus, findEngine, provisionEngine, removeEngines } from './engine.js'

/** One in-flight engine download; a second request joins it instead of racing. */
let installing

export function settingsFile(config) {
  return config.stateFile || path.join(process.env.DSH_HOME || path.join(os.homedir(), '.dsh'), 'local-model', `${config.provider.trim()}.json`)
}

export class ModelSettings {
  constructor(file, config) {
    this.file = file
    this.config = config
    this.revision = 0
    this.models = config.models
    try {
      const saved = JSON.parse(fs.readFileSync(file, 'utf8'))
      if (saved.version !== 1 || !Array.isArray(saved.models) || !Number.isSafeInteger(saved.revision) || saved.revision < 0) throw new Error('invalid settings format')
      this.models = this.validate(saved.models)
      this.revision = saved.revision
    } catch (error) {
      if (error.code !== 'ENOENT') throw new Error(`Cannot read local-model settings at ${file}: ${error.message}`)
    }
  }

  validate(models) {
    if (!Array.isArray(models) || models.length > 200) throw new Error('models 必须是数组，且不超过 200 项')
    const validated = Config({ ...this.config, models }).models
    normalizeModels({ ...this.config, models: validated })
    return validated
  }

  view() {
    return { revision: this.revision, models: this.models.map(({ apiKey, ...model }) => ({ ...model, apiKeyConfigured: Boolean(apiKey) })) }
  }

  mergeSecrets(models) {
    if (!Array.isArray(models)) throw new Error('models 必须是数组')
    return models.map(({ apiKeyConfigured, ...model }) => ({
      ...model,
      apiKey: model.apiKey === undefined ? this.models.find((old) => old.id === model.id)?.apiKey ?? '' : model.apiKey,
    }))
  }

  save(models, revision) {
    if (revision !== this.revision) {
      const error = new Error('设置已被其他页面修改，请刷新后重试')
      error.status = 409
      throw error
    }
    const validated = this.validate(this.mergeSecrets(models))
    fs.mkdirSync(path.dirname(this.file), { recursive: true })
    const temp = `${this.file}.${randomUUID()}.tmp`
    try {
      fs.writeFileSync(temp, JSON.stringify({ version: 1, revision: revision + 1, models: validated }, null, 2), { mode: 0o600, flag: 'wx' })
      fs.renameSync(temp, this.file)
    } finally {
      try { fs.unlinkSync(temp) } catch (error) { if (error.code !== 'ENOENT') throw error }
    }
    this.models = validated
    this.revision += 1
  }
}

export async function browseGguf(directory) {
  if (!directory) {
    const candidates = process.platform === 'win32'
      ? 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').map((drive) => `${drive}:\\`)
      : ['/']
    return { directory: '', parent: null, entries: candidates.filter((file) => fs.existsSync(file)).map((file) => ({ name: file, path: file, directory: true })) }
  }
  if (typeof directory !== 'string' || !path.isAbsolute(directory)) throw new Error('请输入 DSH 所在电脑上的绝对目录路径')
  const resolved = path.resolve(directory)
  const all = await fs.promises.readdir(resolved, { withFileTypes: true })
  const filtered = all.filter((entry) => !entry.name.startsWith('.') && (entry.isDirectory() || entry.isFile() && /\.gguf$/i.test(entry.name)))
  filtered.sort((a, b) => Number(b.isDirectory()) - Number(a.isDirectory()) || a.name.localeCompare(b.name))
  return { directory: resolved, parent: path.dirname(resolved) === resolved ? '' : path.dirname(resolved), truncated: filtered.length > 500,
    entries: filtered.slice(0, 500).map((entry) => ({ name: entry.name, path: path.join(resolved, entry.name), directory: entry.isDirectory() })) }
}

export async function discoverModels(model, { fetchImpl = fetch, signal } = {}) {
  if (model.backend === 'llamacpp') throw new Error('GGUF 模式请直接选择模型文件')
  const url = model.backend === 'ollama' ? `${ollamaRoot(model.baseURL)}/api/tags`
    : chatCompletionsUrl(model.baseURL).replace(/\/chat\/completions$/, '/models')
  const key = model.apiKey || (model.apiKeyEnv ? process.env[model.apiKeyEnv] : '')
  const response = await fetchImpl(url, { headers: key ? { authorization: `Bearer ${key}` } : {},
    redirect: 'error', signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(15_000)]) : AbortSignal.timeout(15_000) })
  if (!response.ok) throw new Error(`模型列表请求失败：HTTP ${response.status}`)
  const raw = await response.json()
  const list = model.backend === 'ollama' ? raw.models : raw.data
  if (!Array.isArray(list)) throw new Error('服务没有返回有效的模型列表；可手动填写模型名')
  return list.map((entry) => ({ id: entry.id ?? entry.name ?? entry.model })).filter((entry) => typeof entry.id === 'string' && entry.id).slice(0, 1000)
}

/**
 * Read the facts behind one draft model: GGUF header or Ollama `/api/show`.
 *
 * The answer feeds the "use this context" affordance in the settings page, so
 * it reports both what the file/model was trained for and what the server would
 * actually allocate. Neither is applied automatically.
 * @param {Record<string, any>} body - `{ file }` or `{ baseURL, remoteModel }`.
 * @param {{ fetchImpl?: typeof fetch }} [options] - transport.
 * @returns {Promise<Record<string, any>>} model facts plus `suggestedContext`.
 * @throws {Error} with an actionable message when the input is unusable.
 */
export async function inspectModel(body, options = {}) {
  const file = typeof body?.file === 'string' ? body.file.trim() : ''
  if (file !== '') {
    if (!path.isAbsolute(file) || !/\.gguf$/i.test(file)) throw new Error('请选择绝对路径的 GGUF 文件')
    const info = await readGgufMetadata(file)
    const warning = mmprojWarning(file, info)
    const server = resolveLlamaServer()
    return {
      kind: warning === undefined ? 'gguf' : 'mmproj',
      ...(warning === undefined ? {} : { warning }),
      ...info,
      ...(server === undefined ? {} : { llamaServer: server }),
      suggestedContext: Math.max(2048, Math.min(info.contextLength ?? 8192, 32_768)),
    }
  }
  const baseURL = typeof body?.baseURL === 'string' ? body.baseURL.trim() : ''
  const remoteModel = typeof body?.remoteModel === 'string' ? body.remoteModel.trim() : ''
  if (baseURL === '' || remoteModel === '') throw new Error('请先填写服务地址和模型名')
  return { kind: 'ollama', ...(await inspectOllama(baseURL, remoteModel, { fetchImpl: options.fetchImpl })) }
}

/**
 * Probe one model's transport without loading it into memory.
 * @param {Record<string, any>} model - a normalized model def.
 * @param {{ fetchImpl?: typeof fetch }} [options] - transport.
 * @returns {Promise<{ ok: boolean, detail: string }>} a human-readable verdict.
 * @throws {Error} when the probe found a blocking problem.
 */
export async function probeModel(model, options = {}) {
  const fetchImpl = options.fetchImpl ?? fetch
  if (model.backend === 'llamacpp') {
    const projection = mmprojWarning(model.file)
    if (projection !== undefined) throw new Error(projection)
    if (!fs.existsSync(model.file)) throw new Error(`GGUF 文件不存在：${model.file}`)
    const info = await readGgufMetadata(model.file)
    const binary = resolveLlamaServer(model.llamaServer) ?? (await findEngine())?.binary
    return {
      ok: true,
      detail: `GGUF 可读，声明上下文 ${info.contextLength ?? '未知'}`
        + `；llama-server ${binary === undefined ? '未找到（可在本页下载内置引擎，或填路径）' : `已找到：${binary}`}`,
    }
  }
  if (model.backend === 'ollama') {
    const result = await detectOllama(model.baseURL, { fetchImpl })
    if (!result.reachable) throw new Error(`Ollama 不可达：${result.error ?? '连接失败'}`)
    const hit = result.models.find((entry) => entry.name === model.remoteModel)
    return {
      ok: true,
      detail: hit === undefined
        ? `已连接 Ollama，但模型列表里没有 ${model.remoteModel}（共 ${result.models.length} 个模型）`
        : `已连接 Ollama，模型存在：${hit.name}${hit.parameterSize ? ` · ${hit.parameterSize}` : ''}`,
    }
  }
  const url = chatCompletionsUrl(model.baseURL).replace(/\/chat\/completions$/, '/models')
  const key = model.apiKey || (model.apiKeyEnv ? process.env[model.apiKeyEnv] : '')
  const response = await fetchImpl(url, {
    headers: key ? { authorization: `Bearer ${key}` } : {},
    redirect: 'error',
    signal: AbortSignal.timeout(10_000),
  })
  if (!response.ok) throw new Error(`模型列表请求失败：HTTP ${response.status}`)
  const raw = await response.json()
  const ids = (Array.isArray(raw?.data) ? raw.data : []).map((entry) => entry?.id).filter((id) => typeof id === 'string')
  return {
    ok: true,
    detail: ids.includes(model.remoteModel)
      ? `已连接，模型存在：${model.remoteModel}`
      : `已连接，模型列表：${ids.slice(0, 8).join(', ') || '（空）'}`,
  }
}

function admission(req, connection) {
  try {
    if (typeof connection?.admit === 'function') {
      const result = connection.admit(req)
      if (result?.then) return 503
      if (typeof result === 'number') return result
      return result?.rejection
    }
    if (typeof connection?.requestRejection === 'function') return connection.requestRejection(req)
  } catch { return 503 }
  return 503
}

async function jsonBody(req) {
  if (!/^application\/json(?:;|$)/i.test(req.headers['content-type'] ?? '')) throw new Error('请求必须使用 application/json')
  let size = 0
  const chunks = []
  for await (const chunk of req) {
    size += chunk.length
    if (size > 1024 * 1024) throw new Error('设置数据超过 1 MiB')
    chunks.push(chunk)
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8'))
}

export function createSettingsApi({ store, connection, save, running = () => [] }) {
  return async (req, res) => {
    const send = (status, value) => {
      if (res.destroyed || res.writableEnded) return
      res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' })
      res.end(JSON.stringify(value))
    }
    const rejection = admission(req, connection())
    if (rejection !== undefined) return send(rejection, { error: 'DSH 身份验证未通过或不可用' })
    // Prefix routes bypass the stock /api handler. Preserve its CSRF boundary.
    if (req.headers['sec-fetch-site'] === 'cross-site') return send(403, { error: '不允许跨站请求' })
    if (req.headers.origin) {
      try {
        const origin = new URL(req.headers.origin)
        if (origin.host !== req.headers.host || !['http:', 'https:'].includes(origin.protocol)) return send(403, { error: '请求来源不匹配' })
      } catch { return send(403, { error: '请求来源无效' }) }
    }
    try {
      const url = new URL(req.url, 'http://localhost')
      const route = url.pathname.replace(/^\/api\/local-model/, '')
      if (req.method === 'GET' && route === '/settings') return send(200, { ...store.view(), running: running() })
      if (req.method === 'GET' && route === '/browse') return send(200, await browseGguf(url.searchParams.get('directory') ?? ''))
      if (req.method === 'GET' && route === '/detect') {
        const baseURL = url.searchParams.get('baseURL') || 'http://127.0.0.1:11434'
        return send(200, { baseURL, ...(await detectOllama(baseURL)) })
      }
      if (req.method === 'GET' && route === '/engine') {
        const installed = (await findEngine()) ?? null
        return send(200, {
          status: engineStatus(),
          installed,
          // What a GGUF request would actually spawn: an explicit/PATH/common
          // install wins over the downloaded one, exactly like the spawner.
          resolved: resolveLlamaServer() ?? installed?.binary ?? null,
        })
      }
      if (req.method !== 'POST') return send(404, { error: '接口不存在' })
      const body = await jsonBody(req)
      if (route === '/engine/install') {
        if (installing === undefined) {
          installing = provisionEngine({ variant: body.variant ?? 'auto' })
            .catch(() => {})
            .finally(() => { installing = undefined })
        }
        return send(200, { started: true, status: engineStatus() })
      }
      if (route === '/engine/remove') {
        await removeEngines()
        return send(200, { status: engineStatus(), installed: null })
      }
      if (route === '/settings') {
        await save(body.models, body.revision)
        return send(200, store.view())
      }
      if (route === '/inspect') return send(200, await inspectModel(body))
      if (route === '/discover') {
        const models = store.validate(store.mergeSecrets([body.model]))
        const [model] = normalizeModels({ ...store.config, models })
        return send(200, { models: await discoverModels(model) })
      }
      if (route === '/test') {
        const models = store.validate(store.mergeSecrets([body.model]))
        const [model] = normalizeModels({ ...store.config, models })
        return send(200, await probeModel(model))
      }
      return send(404, { error: '接口不存在' })
    } catch (error) {
      // Never echo provider responses, URLs, or credentials to the browser.
      send(error.status ?? (error.code === 'MODEL_BUSY' ? 409 : 400), { error: error.message })
    }
  }
}

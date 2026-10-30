/** Native Ollama transport: num_ctx must go to /api/chat, not /v1. */
import { LlmError } from '@deepseek-ai/dsh-llm'
import { buildChatRequest } from './openai.js'

export function ollamaRoot(baseURL) {
  return baseURL.replace(/\/+$/, '').replace(/\/(?:v1(?:\/chat\/completions)?|api(?:\/chat)?)$/, '')
}

export function buildOllamaRequest(options, model, defaults = {}) {
  const wire = buildChatRequest(options, model, defaults)
  const names = new Map()
  const messages = wire.messages.map((message) => {
    const result = { ...message, content: message.content ?? '' }
    if (message.tool_calls) result.tool_calls = message.tool_calls.map((call) => {
      names.set(call.id, call.function.name)
      let args
      try { args = JSON.parse(call.function.arguments) } catch {
        throw new LlmError('Ollama tool history contains invalid JSON arguments', 'INVALID_REQUEST')
      }
      return { ...call, function: { ...call.function, arguments: args } }
    })
    if (message.role === 'tool') result.tool_name = names.get(message.tool_call_id) ?? ''
    return result
  })
  const extra = model.extraBody ?? {}
  return {
    ...extra,
    model: wire.model, messages, stream: true,
    // `think` is top-level for /api/chat (not inside `options`). Only an
    // explicit opt-out is sent, so the model keeps its own default otherwise.
    ...(model.think === false ? { think: false } : {}),
    ...(wire.tools ? { tools: wire.tools } : {}),
    options: {
      ...(extra.options ?? {}),
      ...(wire.max_tokens ? { num_predict: wire.max_tokens } : {}),
      ...(wire.temperature === undefined ? {} : { temperature: wire.temperature }),
      ...(wire.stop ? { stop: wire.stop } : {}),
      ...(model.contextWindow ? { num_ctx: model.contextWindow } : {}),
    },
  }
}

/** Convert Ollama NDJSON into the shared completion translator's event shape. */
export async function* readOllama(body, signal, state = {}) {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let nextTool = 0
  const convert = (line) => {
    let raw
    try { raw = JSON.parse(line) } catch {
      throw new LlmError('Ollama returned invalid NDJSON', 'STREAM_CLOSED')
    }
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
      throw new LlmError('Ollama returned an invalid event', 'STREAM_CLOSED')
    }
    if (raw.error) return { error: raw.error }
    const message = raw.message ?? {}
    const toolCalls = message.tool_calls?.map((call) => {
      const index = nextTool++
      return { ...call, index, id: call.id || `ollama_call_${index}` }
    })
    if (raw.done) state.sawDone = true
    return {
      choices: [{ delta: {
        content: message.content, reasoning: message.thinking,
        ...(toolCalls ? { tool_calls: toolCalls } : {}),
      }, ...(raw.done ? { finish_reason: raw.done_reason === 'length' ? 'length' : 'stop' } : {}) }],
      ...(raw.done ? { usage: {
        prompt_tokens: raw.prompt_eval_count, completion_tokens: raw.eval_count,
        prompt_tokens_details: { cached_tokens: raw.prompt_eval_cached_count ?? 0 },
      } } : {}),
    }
  }
  try {
    while (true) {
      signal?.throwIfAborted()
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      if (buffer.length > 8 * 1024 * 1024) throw new LlmError('Ollama event too large', 'STREAM_CLOSED')
      let end
      while ((end = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, end).trim()
        buffer = buffer.slice(end + 1)
        if (line) yield convert(line)
        if (state.sawDone) return
      }
    }
    buffer += decoder.decode()
    if (buffer.trim()) yield convert(buffer.trim())
  } finally {
    await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}

/**
 * Parse Ollama's Modelfile-style `parameters` text into plain values.
 * @param {unknown} text - lines such as `num_ctx 32768`.
 * @returns {Record<string, string | number>} one entry per parameter.
 */
export function parseOllamaParameters(text) {
  /** @type {Record<string, string | number>} */
  const values = {}
  if (typeof text !== 'string') return values
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (trimmed === '') continue
    const match = /^([A-Za-z0-9_.-]+)\s+(.+)$/.exec(trimmed)
    if (match === null) continue
    const raw = match[2].trim()
    values[match[1]] = /^-?\d+(?:\.\d+)?$/.test(raw) ? Number(raw) : raw
  }
  return values
}

/**
 * Read one Ollama model's own facts from `/api/show`.
 *
 * The interesting number is not always the trained context: a Modelfile can
 * pin `num_ctx`, and that pinned value is what the server will actually
 * allocate. `suggestedContext` prefers the pinned value, then the trained one,
 * capped at 32K so a settings click cannot accidentally ask for a 256K KV cache.
 *
 * @param {string} baseURL - Ollama root (with or without `/v1`).
 * @param {string} model - the model name/tag.
 * @param {{ fetchImpl?: typeof fetch, signal?: AbortSignal, timeoutMs?: number }} [options] - transport.
 * @returns {Promise<Record<string, any>>} model facts.
 * @throws {Error} when the model is unknown or the server refuses the request.
 */
export async function inspectOllama(baseURL, model, options = {}) {
  const fetchImpl = options.fetchImpl ?? fetch
  const timeoutMs = options.timeoutMs ?? 15_000
  const signal = options.signal
    ? AbortSignal.any([options.signal, AbortSignal.timeout(timeoutMs)])
    : AbortSignal.timeout(timeoutMs)
  const response = await fetchImpl(`${ollamaRoot(baseURL)}/api/show`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model }),
    redirect: 'error',
    signal,
  })
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new Error(`Ollama /api/show 返回 HTTP ${response.status}${text ? `：${text.slice(0, 200)}` : ''}`)
  }
  const raw = await response.json()
  const info = raw?.model_info && typeof raw.model_info === 'object' ? raw.model_info : {}
  const architecture = typeof info['general.architecture'] === 'string'
    ? info['general.architecture']
    : Object.keys(info).map((key) => /^([^.]+)\.context_length$/.exec(key)?.[1]).find(Boolean)
  const trained = typeof info[`${architecture}.context_length`] === 'number'
    ? info[`${architecture}.context_length`]
    : Object.entries(info).find(([key]) => key.endsWith('.context_length'))?.[1]
  const parameters = parseOllamaParameters(raw?.parameters)
  const pinned = typeof parameters.num_ctx === 'number' && parameters.num_ctx > 0 ? parameters.num_ctx : undefined
  const effective = pinned ?? (typeof trained === 'number' && trained > 0 ? trained : undefined)
  return {
    model,
    family: raw?.details?.family,
    parameterSize: raw?.details?.parameter_size,
    quantization: raw?.details?.quantization_level,
    capabilities: Array.isArray(raw?.capabilities) ? raw.capabilities : [],
    architecture,
    trainedContext: typeof trained === 'number' ? trained : undefined,
    configuredContext: pinned,
    suggestedContext: effective === undefined ? 8192 : Math.max(2048, Math.min(effective, 32_768)),
    parameters,
  }
}

/**
 * Probe a local Ollama server and list the models it can serve.
 * @param {string} baseURL - Ollama root.
 * @param {{ fetchImpl?: typeof fetch, signal?: AbortSignal, timeoutMs?: number }} [options] - transport.
 * @returns {Promise<{ reachable: boolean, models: Array<Record<string, any>>, error?: string }>} the probe result.
 */
export async function detectOllama(baseURL, options = {}) {
  const fetchImpl = options.fetchImpl ?? fetch
  const timeoutMs = options.timeoutMs ?? 5_000
  const signal = options.signal
    ? AbortSignal.any([options.signal, AbortSignal.timeout(timeoutMs)])
    : AbortSignal.timeout(timeoutMs)
  try {
    const response = await fetchImpl(`${ollamaRoot(baseURL)}/api/tags`, { redirect: 'error', signal })
    if (!response.ok) return { reachable: false, models: [], error: `HTTP ${response.status}` }
    const raw = await response.json()
    const models = (Array.isArray(raw?.models) ? raw.models : []).slice(0, 200).map((entry) => ({
      name: String(entry?.name ?? entry?.model ?? ''),
      size: typeof entry?.size === 'number' ? entry.size : undefined,
      parameterSize: entry?.details?.parameter_size,
      quantization: entry?.details?.quantization_level,
      family: entry?.details?.family,
      modifiedAt: entry?.modified_at,
    })).filter((entry) => entry.name !== '')
    return { reachable: true, models }
  } catch (error) {
    return { reachable: false, models: [], error: error instanceof Error ? error.message : String(error) }
  }
}

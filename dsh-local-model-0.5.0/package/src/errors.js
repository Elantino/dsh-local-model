/**
 * Failure construction and HTTP-status classification for the local-model
 * provider.
 *
 * DSH routes on `LlmError#code` and never parses message text, so every failure
 * this adapter can raise is built here. One place keeps the taxonomy aligned
 * with the rest of the harness (`AUTH`, `RATE_LIMIT`, `SERVER`, `TRANSPORT`,
 * `TIMEOUT`, `INVALID_REQUEST`, `UNKNOWN_MODEL`, `CONTEXT_WINDOW_EXCEEDED`,
 * `ABORTED`) and keeps provider-supplied text out of control flow.
 *
 * @module dsh-local-model/errors
 */

import { LlmError, isContextWindowExceededError } from '@deepseek-ai/dsh-llm'

/** No llama-server binary could be resolved, or it could not be started. */
export const LOCAL_RUNTIME_MISSING = 'LOCAL_RUNTIME_MISSING'

/** A llama-server process started and then failed before serving a request. */
export const LOCAL_RUNTIME_FAILED = 'LOCAL_RUNTIME_FAILED'

/** The event stream ended without a terminal marker. */
export const STREAM_CLOSED = 'STREAM_CLOSED'

/** The provider reported an error inside an otherwise healthy SSE stream. */
export const PROVIDER_ERROR = 'PROVIDER_ERROR'

/**
 * Read the provider's own sentence out of an error body without trusting it.
 *
 * Local servers answer with a surprising variety of shapes — OpenAI's
 * `{"error":{"message":…}}`, llama.cpp's `{"error":{"code":…,"message":…}}`,
 * Ollama's `{"error":"…"}` — so this walks the shapes we know and otherwise
 * collapses the raw text. The result is only ever used for display.
 * @param {unknown} body - decoded JSON body or raw response text.
 * @returns {string} a single-line, length-bounded diagnostic.
 */
export function providerDetail(body) {
  let raw = body
  if (typeof raw === 'string') {
    const text = raw.trim()
    if (text === '') return ''
    try {
      raw = JSON.parse(text)
    } catch {
      return collapse(text)
    }
  }
  if (typeof raw === 'object' && raw !== null) {
    const error = /** @type {Record<string, unknown>} */ (raw).error
    if (typeof error === 'string') return collapse(error)
    if (typeof error === 'object' && error !== null) {
      const fields = /** @type {Record<string, unknown>} */ (error)
      const parts = [fields.message, fields.code, fields.type]
        .filter((value) => typeof value === 'string' && value !== '')
      if (parts.length > 0) return collapse(parts.join(' '))
    }
    const message = /** @type {Record<string, unknown>} */ (raw).message
    if (typeof message === 'string') return collapse(message)
  }
  return ''
}

/**
 * Flatten whitespace and bound the length of a provider sentence.
 * @param {string} text - arbitrary provider text.
 * @returns {string} one bounded line.
 */
function collapse(text) {
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length > 300 ? `${flat.slice(0, 300)}…` : flat
}

/**
 * Classify one HTTP status plus its provider detail into a stable code.
 * @param {number} status - HTTP status the server returned.
 * @param {string} detail - provider detail from {@link providerDetail}.
 * @returns {string} a stable provider-neutral failure code.
 */
function classify(status, detail) {
  if (detail !== '' && isContextWindowExceededError(detail)) return 'CONTEXT_WINDOW_EXCEEDED'
  if (status === 401 || status === 403) return 'AUTH'
  if (status === 404 && /\bmodel\b/i.test(detail)) return 'UNKNOWN_MODEL'
  if (status === 404) return 'INVALID_REQUEST'
  if (status === 429) return 'RATE_LIMIT'
  if (status === 408 || status === 504) return 'TIMEOUT'
  if (status === 400 || status === 413 || status === 422) return 'INVALID_REQUEST'
  if (status >= 500) return 'SERVER'
  return 'TRANSPORT'
}

/**
 * Build the failure for a non-2xx response.
 * @param {number} status - HTTP status the server returned.
 * @param {unknown} body - decoded body or raw response text.
 * @param {string} where - endpoint description used in the message.
 * @returns {LlmError} a terminal failure carrying the HTTP status.
 */
export function httpFailure(status, body, where) {
  const detail = providerDetail(body)
  const overflow = /request \((\d+) tokens?\) exceeds the available context size \((\d+) tokens?\)/i.exec(detail)
  if (overflow !== null) {
    const requested = Number(overflow[1])
    const available = Number(overflow[2])
    return new LlmError(
      `${where} 上下文不足：本次请求 ${requested} tokens，引擎上限 ${available} tokens。`
        + `在“本地模型与 API”里把该模型的上下文调到 ${Math.max(32_768, Math.ceil(requested / 4096) * 4096)} 以上，`
        + '并把 KV 缓存设为 q8_0、Flash Attention 设为 on（8GB 显存也能跑 32K）；或者先压缩当前会话。',
      'CONTEXT_WINDOW_EXCEEDED',
      { status },
    )
  }
  const message = detail === ''
    ? `${where} failed with HTTP ${status}`
    : `${where} failed with HTTP ${status}: ${detail}`
  return new LlmError(message, classify(status, detail), { status })
}

/**
 * Build the failure for an error object delivered inside a 200 SSE stream.
 * @param {unknown} error - the `error` member of an in-band event.
 * @param {string} where - endpoint description used in the message.
 * @returns {LlmError} a terminal failure without an HTTP status.
 */
export function inBandFailure(error, where) {
  const detail = providerDetail({ error })
  const message = detail === ''
    ? `${where} reported an error inside the response stream`
    : `${where} reported an error inside the response stream: ${detail}`
  return new LlmError(message, PROVIDER_ERROR)
}

/**
 * Build the failure for a request the caller cancelled.
 * @param {string} where - endpoint description used in the message.
 * @returns {LlmError} an `ABORTED` failure.
 */
export function abortedFailure(where) {
  return new LlmError(`${where} request was aborted`, 'ABORTED')
}

/**
 * Build the failure for a transport-level rejection (connection refused, DNS,
 * socket reset, timeout signal).
 * @param {unknown} cause - the thrown transport error.
 * @param {string} where - endpoint description used in the message.
 * @returns {LlmError} a `TRANSPORT` failure that keeps the cause chain.
 */
export function transportFailure(cause, where) {
  const reason = cause instanceof Error ? cause.message : String(cause)
  return new LlmError(`${where} could not be reached: ${reason}`, 'TRANSPORT', { cause })
}

/**
 * Build the failure for an unresolvable or unstartable llama-server binary.
 * @param {string} binary - the path or name that was tried.
 * @param {unknown} [cause] - the spawn rejection, when one happened.
 * @returns {LlmError} a terminal configuration failure with repair steps.
 */
export function missingRuntimeFailure(binary, cause) {
  return new LlmError(
    `local-model: cannot start llama-server ("${binary}"). Install llama.cpp and put llama-server on PATH, `
      + 'set `llamaServer` to its absolute path, or set `backend: openai` on the model to talk to a server '
      + 'that is already running (Ollama, LM Studio, llama-server). See https://github.com/ggerganov/llama.cpp',
    LOCAL_RUNTIME_MISSING,
    cause === undefined ? undefined : { cause },
  )
}

/**
 * Build the failure for a llama-server that started but never became healthy.
 * @param {string} reason - short account of what happened.
 * @param {string} stderr - captured process output tail.
 * @returns {LlmError} a `LOCAL_RUNTIME_FAILED` failure including the tail.
 */
export function runtimeFailure(reason, stderr) {
  const tail = typeof stderr === 'string' ? stderr.trim() : ''
  return new LlmError(
    tail === ''
      ? `local-model: ${reason}`
      : `local-model: ${reason}; last output: ${collapse(tail)}`,
    LOCAL_RUNTIME_FAILED,
  )
}

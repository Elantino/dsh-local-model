/**
 * A small Server-Sent Events reader for OpenAI-compatible chat completions.
 *
 * The reader is deliberately tolerant: local servers differ in line endings,
 * comment lines, whether they send `data: [DONE]`, and whether they put one
 * JSON object or several per network chunk. It yields one decoded JSON value
 * per event and reports whether the terminal `[DONE]` marker was seen, so the
 * translator can distinguish a graceful close from a truncated stream.
 *
 * @module dsh-local-model/sse
 */

import { LlmError } from '@deepseek-ai/dsh-llm'
import { STREAM_CLOSED } from './errors.js'

/**
 * Parse an SSE response body into JSON events.
 * @param {ReadableStream<Uint8Array>} body - the response body stream.
 * @param {AbortSignal | undefined} signal - caller cancellation.
 * @param {{ sawDone?: boolean }} [state] - receives `sawDone: true` when the
 *   server sent the `[DONE]` sentinel.
 * @returns {AsyncGenerator<Record<string, unknown>, void, unknown>} decoded events.
 * @throws {LlmError} `STREAM_CLOSED` when an event payload is not a JSON object.
 */
export async function* readSse(body, signal, state = {}) {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  /** @type {string[]} */
  let data = []

  /**
   * Join the buffered `data:` lines of one event.
   * @returns {string | undefined} the payload, or undefined for an empty event.
   */
  const takePayload = () => {
    if (data.length === 0) return undefined
    const payload = data.join('\n')
    data = []
    return payload
  }

  /**
   * Decode one JSON event payload.
   * @param {string} payload - the event payload.
   * @returns {Record<string, unknown>} the decoded event object.
   */
  const decode = (payload) => {
    try {
      const value = JSON.parse(payload)
      if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        throw new LlmError('local-model: stream event was not a JSON object', STREAM_CLOSED)
      }
      return /** @type {Record<string, unknown>} */ (value)
    } catch (error) {
      if (error instanceof LlmError) throw error
      throw new LlmError(
        `local-model: could not decode a stream event: ${payload.slice(0, 200)}`,
        STREAM_CLOSED,
        { cause: error },
      )
    }
  }

  try {
    while (true) {
      signal?.throwIfAborted()
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      if (buffer.length + data.reduce((size, line) => size + line.length, 0) > 8 * 1024 * 1024) {
        throw new LlmError('local-model: stream event exceeds 8 MiB', STREAM_CLOSED)
      }
      let newline = buffer.indexOf('\n')
      while (newline !== -1) {
        let line = buffer.slice(0, newline)
        buffer = buffer.slice(newline + 1)
        newline = buffer.indexOf('\n')
        if (line.endsWith('\r')) line = line.slice(0, -1)
        if (line === '') {
          const payload = takePayload()
          if (payload === undefined) continue
          if (payload === '[DONE]') {
            state.sawDone = true
            return
          }
          yield decode(payload)
          continue
        }
        if (line.startsWith(':')) continue
        if (!line.startsWith('data:')) continue
        let chunk = line.slice(5)
        if (chunk.startsWith(' ')) chunk = chunk.slice(1)
        data.push(chunk)
      }
    }
    buffer += decoder.decode()
    // A final event whose terminating blank line was never sent.
    if (buffer !== '') {
      if (buffer.endsWith('\r')) buffer = buffer.slice(0, -1)
      if (buffer.startsWith('data:')) {
        let chunk = buffer.slice(5)
        if (chunk.startsWith(' ')) chunk = chunk.slice(1)
        data.push(chunk)
      }
    }
    const payload = takePayload()
    if (payload === undefined) return
    if (payload === '[DONE]') {
      state.sawDone = true
      return
    }
    yield decode(payload)
  } finally {
    try {
      await reader.cancel()
    } catch {
      // The body is already closing; a failed cancel is not a request failure.
    }
    reader.releaseLock()
  }
}

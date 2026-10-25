/**
 * Translate an OpenAI-compatible chat-completions event stream into the
 * harness `StreamChunk` protocol.
 *
 * The protocol's rules this file enforces:
 *
 * - blocks get dense, creation-ordered indexes;
 * - every `block-start` is eventually closed by exactly one `block-end`;
 * - `tool-call-delta` carries the id from the first fragment onward, so a
 *   consumer never has to guess which call it belongs to;
 * - `usage` precedes the single terminal `finish`;
 * - nothing is emitted after `finish`.
 *
 * Reasoning is read from both spellings local servers use
 * (`delta.reasoning_content` from llama.cpp/DeepSeek-derived templates and
 * `delta.reasoning` from several Ollama builds) and becomes a `reasoning`
 * block rather than leaking into visible text.
 *
 * @module dsh-local-model/translate
 */

import { EMPTY_RESPONSE_CODE, LlmError } from '@deepseek-ai/dsh-llm'
import { STREAM_CLOSED, inBandFailure } from './errors.js'
import { usageFrom, usageFromTimings } from './openai.js'

/**
 * Map one provider finish reason onto the harness reason.
 *
 * A provider that says `tool_calls` without ever streaming one is corrected to
 * `stop`: the harness would otherwise look for calls it never received.
 * @param {string | undefined} reason - the provider's `finish_reason`.
 * @param {number} toolCalls - how many tool calls the response carried.
 * @returns {{ kind: 'stop' } | { kind: 'tool-calls' } | { kind: 'max-tokens' }} the mapped reason.
 */
function finishFor(reason, toolCalls) {
  if (reason === 'length') return { kind: 'max-tokens' }
  if (toolCalls > 0) return { kind: 'tool-calls' }
  return { kind: 'stop' }
}

/**
 * Yield harness chunks for one provider response.
 * @param {AsyncIterable<Record<string, any>>} events - decoded SSE events.
 * @param {{ where?: string, state?: { sawDone?: boolean } }} [options] - endpoint label
 *   and the reader state that reports the terminal `[DONE]` marker.
 * @returns {AsyncGenerator<Record<string, any>, void, unknown>} harness stream chunks.
 * @throws {LlmError} on an in-band provider error, a truncated stream, or an empty completion.
 */
export async function* translateChatCompletion(events, options = {}) {
  const where = options.where ?? 'local model endpoint'
  const state = options.state ?? {}
  /** @type {number} */
  let nextIndex = 0
  /** @type {{ index: number, type: 'text' | 'reasoning', text: string } | undefined} */
  let open
  /** @type {Map<number, { index: number, id: string, name: string, arguments: string }>} */
  const toolCalls = new Map()
  /** @type {Record<string, number> | undefined} */
  let usage
  /** @type {string | undefined} */
  let finishReason
  let emitted = 0

  /** Close the currently open text/reasoning block, if any. */
  function* closeOpen() {
    if (open === undefined) return
    const index = open.index
    const block = open.type === 'text'
      ? { type: 'text', text: open.text }
      : { type: 'reasoning', text: open.text }
    open = undefined
    yield { type: 'block-end', index, block }
  }

  /**
   * Ensure a block of one type is open, switching blocks when the type changes.
   * @param {'text' | 'reasoning'} type - the requested block type.
   */
  function* ensure(type) {
    if (open !== undefined && open.type === type) return
    yield* closeOpen()
    open = { index: nextIndex, type, text: '' }
    nextIndex += 1
    emitted += 1
    yield { type: 'block-start', index: open.index, blockType: type }
  }

  for await (const event of events) {
    if (event.error !== undefined) throw inBandFailure(event.error, where)
    if (event.usage !== undefined) usage = usageFrom(event.usage) ?? usage
    if (usage === undefined && event.timings !== undefined) usage = usageFromTimings(event.timings)

    const choice = Array.isArray(event.choices) ? event.choices[0] : undefined
    if (choice === undefined || choice === null) continue
    const delta = choice.delta ?? {}

    const reasoning = typeof delta.reasoning_content === 'string'
      ? delta.reasoning_content
      : typeof delta.reasoning === 'string' ? delta.reasoning : ''
    if (reasoning !== '') {
      yield* ensure('reasoning')
      open.text += reasoning
      yield { type: 'reasoning-delta', index: open.index, text: reasoning }
    }

    const content = typeof delta.content === 'string' ? delta.content : ''
    if (content !== '') {
      yield* ensure('text')
      open.text += content
      yield { type: 'text-delta', index: open.index, text: content }
    }

    if (Array.isArray(delta.tool_calls)) {
      for (const call of delta.tool_calls) {
        if (typeof call !== 'object' || call === null) continue
        const slot = Number.isInteger(call.index) ? call.index : 0
        let entry = toolCalls.get(slot)
        if (entry === undefined) {
          yield* closeOpen()
          entry = { index: nextIndex, id: '', name: '', arguments: '' }
          nextIndex += 1
          emitted += 1
          toolCalls.set(slot, entry)
          yield { type: 'block-start', index: entry.index, blockType: 'tool-call' }
        }
        if (entry.id === '' && typeof call.id === 'string' && call.id !== '') entry.id = call.id
        const fn = typeof call.function === 'object' && call.function !== null ? call.function : {}
        if (typeof fn.name === 'string' && fn.name !== '') entry.name = fn.name
        let fragment = ''
        if (typeof fn.arguments === 'string') fragment = fn.arguments
        else if (fn.arguments !== undefined && fn.arguments !== null) fragment = JSON.stringify(fn.arguments)
        if (entry.id === '') entry.id = `local_call_${slot}_${entry.index}`
        entry.arguments += fragment
        if (fragment !== '' || entry.name !== '') {
          yield {
            type: 'tool-call-delta',
            index: entry.index,
            id: entry.id,
            ...(entry.name === '' ? {} : { name: entry.name }),
            argumentsDelta: fragment,
          }
        }
      }
    }

    if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') {
      finishReason = choice.finish_reason
    }
  }

  yield* closeOpen()

  for (const entry of toolCalls.values()) {
    if (entry.name === '') {
      throw new LlmError(`${where} streamed a tool call without a function name`, STREAM_CLOSED)
    }
    yield {
      type: 'block-end',
      index: entry.index,
      block: {
        type: 'tool-call',
        id: entry.id,
        name: entry.name,
        arguments: entry.arguments === '' ? '{}' : entry.arguments,
      },
    }
  }

  if (finishReason === undefined && state.sawDone !== true) {
    throw new LlmError(`${where} closed the stream before sending a finish reason`, STREAM_CLOSED)
  }
  if (emitted === 0) {
    throw new LlmError(`${where} returned a completion with no content`, EMPTY_RESPONSE_CODE)
  }
  if (usage !== undefined) yield { type: 'usage', usage }
  yield { type: 'finish', reason: finishFor(finishReason, toolCalls.size) }
}

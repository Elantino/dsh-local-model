/**
 * Translation between the harness message vocabulary and the OpenAI-compatible
 * chat-completions wire format used by llama.cpp, Ollama, LM Studio and vLLM.
 *
 * Everything here is pure: no I/O, no clocks, no logging. That keeps the wire
 * shape testable without a server and keeps provider quirks in one file.
 *
 * @module dsh-local-model/openai
 */

/**
 * Join every visible text block of one message.
 * @param {ReadonlyArray<any>} blocks - content blocks.
 * @returns {string} the joined text.
 */
export function textOf(blocks) {
  const parts = []
  for (const block of blocks) {
    if (block?.type === 'text' && typeof block.text === 'string' && block.text !== '') parts.push(block.text)
  }
  return parts.join('\n')
}

/**
 * Describe one non-text, non-replayable block as text.
 *
 * The runtime already projects files and images for text-only routes; this is
 * a defensive last resort, because a local chat template that receives an
 * unknown content part is much harder to diagnose than a placeholder line.
 * @param {any} block - the content block.
 * @returns {string} placeholder text.
 */
function placeholderOf(block) {
  if (block?.type === 'image') {
    const name = block.attachment?.name ?? block.attachment?.attachmentId ?? 'image'
    return `[image omitted: ${name}]`
  }
  if (block?.type === 'file') {
    const name = block.attachment?.name ?? block.attachment?.attachmentId ?? 'file'
    return `[file: ${name}]`
  }
  return ''
}

/**
 * Project one assistant message onto the wire format.
 * @param {any} message - the durable assistant message.
 * @param {number} index - position, for diagnostics.
 * @returns {Record<string, unknown> | undefined} the wire message, or undefined when it has no content.
 */
function assistantMessage(message, index) {
  const text = textOf(message.content)
  /** @type {Array<Record<string, unknown>>} */
  const toolCalls = []
  for (const block of message.content) {
    if (block?.type !== 'tool-call') continue
    toolCalls.push({
      id: String(block.id ?? `local_call_${index}_${toolCalls.length}`),
      type: 'function',
      function: {
        name: String(block.name ?? ''),
        arguments: typeof block.arguments === 'string' ? block.arguments : JSON.stringify(block.arguments ?? {}),
      },
    })
  }
  if (text === '' && toolCalls.length === 0) return undefined
  return {
    role: 'assistant',
    content: text === '' ? null : text,
    ...(toolCalls.length === 0 ? {} : { tool_calls: toolCalls }),
  }
}

/**
 * Project the request messages onto the wire format.
 * @param {{ system?: string, messages: ReadonlyArray<any> }} options - the request.
 * @param {Map<string, { data: Uint8Array, mediaType: string }>} [images] - request bytes per
 *   attachment id; an absent entry becomes a text placeholder.
 * @returns {Array<Record<string, unknown>>} wire messages in order.
 */
export function toWireMessages(options, images) {
  /** @type {Array<Record<string, unknown>>} */
  const messages = []
  if (typeof options.system === 'string' && options.system.trim() !== '') {
    messages.push({ role: 'system', content: options.system })
  }
  options.messages.forEach((message, index) => {
    switch (message.role) {
      case 'system': {
        const text = textOf(message.content)
        if (text !== '') messages.push({ role: 'system', content: text })
        return
      }
      case 'developer':
        // The runtime projects developer messages away for routes that do not
        // declare `toolUpdate`; ignoring them here is the same contract.
        return
      case 'user': {
        const text = textOf(message.content)
        /** @type {Array<Record<string, any>>} */
        const parts = []
        for (const block of message.content) {
          if (block?.type === 'text' && block.text !== '') {
            parts.push({ type: 'text', text: block.text })
            continue
          }
          if (block?.type === 'image') {
            const version = block.offloaded === true ? undefined : images?.get(block.attachment?.attachmentId)
            if (version !== undefined) {
              parts.push({
                type: 'image_url',
                image_url: { url: `data:${version.mediaType};base64,${Buffer.from(version.data).toString('base64')}` },
              })
            } else {
              parts.push({ type: 'text', text: placeholderOf(block) })
            }
            continue
          }
          if (block?.type === 'file') {
            const placeholder = placeholderOf(block)
            if (placeholder !== '') parts.push({ type: 'text', text: placeholder })
          }
        }
        if (parts.length === 0) return
        // Text-only content stays a plain string: local templates handle that
        // shape best, and a parts array is only needed to carry an image.
        const withImage = parts.some((part) => part.type === 'image_url')
        messages.push({
          role: 'user',
          content: withImage ? parts : parts.map((part) => part.text).join('\n'),
        })
        return
      }
      case 'assistant': {
        const wire = assistantMessage(message, index)
        if (wire !== undefined) messages.push(wire)
        return
      }
      case 'tool': {
        const text = textOf(message.content)
        messages.push({
          role: 'tool',
          tool_call_id: String(message.toolCallId ?? message.source?.callId ?? ''),
          content: text === '' ? '(no output)' : text,
        })
        return
      }
      default:
        return
    }
  })
  return messages
}

/**
 * Build one chat-completions request body.
 * @param {any} options - the harness request.
 * @param {Record<string, any>} model - the normalized model def.
 * @param {{ defaultMaxTokens?: number, images?: Map<string, { data: Uint8Array, mediaType: string }> }} [defaults]
 *   provider-level fallbacks and the prepared request images.
 * @returns {Record<string, unknown>} the JSON body to POST.
 */
export function buildChatRequest(options, model, defaults = {}) {
  /** @type {Record<string, unknown>} */
  const body = {
    model: model.remoteModel ?? options.model,
    messages: toWireMessages(options, defaults.images),
    stream: true,
    stream_options: { include_usage: true },
  }
  const maxTokens = options.maxTokens ?? model.maxTokens ?? defaults.defaultMaxTokens
  if (typeof maxTokens === 'number' && maxTokens > 0) body.max_tokens = maxTokens
  if (typeof options.temperature === 'number') body.temperature = options.temperature
  if (Array.isArray(options.stop) && options.stop.length > 0) body.stop = [...options.stop]
  if (Array.isArray(options.tools) && options.tools.length > 0) {
    body.tools = options.tools.map((tool) => ({
      type: 'function',
      function: {
        name: String(tool.name),
        description: typeof tool.description === 'string' ? tool.description : '',
        parameters: typeof tool.parameters === 'object' && tool.parameters !== null
          ? tool.parameters
          : { type: 'object', properties: {} },
      },
    }))
    body.tool_choice = 'auto'
  }
  // `extraBody` is applied first so it can add parameters (top_k, repeat_penalty,
  // chat_template_kwargs) without ever being able to desynchronize the stream
  // framing this adapter depends on.
  const merged = { ...(model.extraBody ?? {}), ...body }
  // A managed llama-server gets the thinking switch through its Jinja template.
  // Only the explicit opt-out is sent, and an existing `chat_template_kwargs`
  // from `extraBody` wins, so a user-provided template setting is never
  // silently overwritten. Cloud OpenAI-compatible APIs are left untouched.
  // A managed llama-server and an existing one (`service: 'llama'`) both read
  // the Jinja template kwargs; a cloud OpenAI-compatible API is left untouched
  // because an unknown field there can be a hard 400.
  const jinjaServer = model.backend === 'llamacpp' || model.service === 'llama'
  if (jinjaServer && model.think === false && merged.chat_template_kwargs === undefined) {
    merged.chat_template_kwargs = { enable_thinking: false }
  }
  return merged
}

/**
 * Normalize a token-usage object into the harness vocabulary.
 *
 * Harness counts are disjoint: `inputTokens` excludes cached input, which is
 * reported separately, so a provider that folds cache hits into `prompt_tokens`
 * (as OpenAI-compatible servers do) has them subtracted here.
 * @param {unknown} raw - the provider's `usage` object.
 * @returns {Record<string, number> | undefined} normalized usage, when present.
 */
export function usageFrom(raw) {
  if (typeof raw !== 'object' || raw === null) return undefined
  const usage = /** @type {Record<string, any>} */ (raw)
  const prompt = finite(usage.prompt_tokens)
  const completion = finite(usage.completion_tokens)
  if (prompt === undefined && completion === undefined) return undefined
  const cached = finite(usage.prompt_tokens_details?.cached_tokens) ?? finite(usage.cache_read_input_tokens) ?? 0
  const reasoning = finite(usage.completion_tokens_details?.reasoning_tokens)
  const total = finite(usage.total_tokens)
  return {
    inputTokens: Math.max(0, (prompt ?? 0) - cached),
    outputTokens: completion ?? 0,
    ...(total === undefined ? {} : { totalTokens: total }),
    ...(cached > 0 ? { cacheReadTokens: cached } : {}),
    ...(reasoning === undefined ? {} : { reasoningTokens: reasoning }),
  }
}

/**
 * Derive usage from llama.cpp's non-OpenAI `timings` object.
 *
 * llama.cpp reports timings on the final chunk even when `stream_options`
 * usage accounting is unsupported, so this keeps the token meter working on
 * older builds.
 * @param {unknown} raw - the `timings` object.
 * @returns {Record<string, number> | undefined} normalized usage, when present.
 */
export function usageFromTimings(raw) {
  if (typeof raw !== 'object' || raw === null) return undefined
  const timings = /** @type {Record<string, any>} */ (raw)
  const prompt = finite(timings.prompt_n)
  const predicted = finite(timings.predicted_n)
  if (prompt === undefined && predicted === undefined) return undefined
  return {
    inputTokens: prompt ?? 0,
    outputTokens: predicted ?? 0,
  }
}

/**
 * Build the chat-completions URL from an OpenAI-compatible root.
 *
 * The configured value is a root (`http://host:port/v1`), but pasting a full
 * endpoint is a common mistake, so both forms are accepted.
 * @param {string} baseURL - the configured root or full endpoint.
 * @returns {string} the POST target.
 */
export function chatCompletionsUrl(baseURL) {
  const trimmed = String(baseURL ?? '').trim().replace(/\/+$/, '')
  return /\/chat\/completions$/.test(trimmed) ? trimmed : `${trimmed}/chat/completions`
}

/**
 * Read a finite non-negative number.
 * @param {unknown} value - candidate.
 * @returns {number | undefined} the number, when usable.
 */
function finite(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined
}

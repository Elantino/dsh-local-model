/**
 * The provider-neutral adapter for local models.
 *
 * It owns no wire code itself: message/usage shaping lives in `openai.js`,
 * stream translation in `translate.js`, process lifetime in `llama.js`, and
 * metadata in `models.js`. What remains here is the `LlmAdapter` contract —
 * route metadata, the model catalog, exact-model resolution, and one streaming
 * request per call.
 *
 * `prepareCall` is inherited: it resolves the model once and binds the stream
 * to that resolution, which is exactly right for a provider whose catalog is
 * static config.
 *
 * @module dsh-local-model/adapter
 */

import { LlmAdapter, LlmError, attributionHeaders } from '@deepseek-ai/dsh-llm'

import { ModelRegistry } from './models.js'
import { buildChatRequest, chatCompletionsUrl } from './openai.js'
import { readSse } from './sse.js'
import { translateChatCompletion } from './translate.js'
import { abortedFailure, httpFailure, transportFailure } from './errors.js'
import { buildOllamaRequest, ollamaRoot, readOllama } from './ollama.js'

/**
 * Read the bearer token for one OpenAI-compatible model.
 * @param {Record<string, any>} model - the normalized def.
 * @param {NodeJS.ProcessEnv} env - environment for `apiKeyEnv`.
 * @returns {string} the token, or an empty string.
 */
function resolveApiKey(model, env) {
  if (typeof model.apiKey === 'string' && model.apiKey !== '') return model.apiKey
  if (typeof model.apiKeyEnv === 'string' && model.apiKeyEnv !== '') {
    const value = env[model.apiKeyEnv]
    if (typeof value === 'string' && value.trim() !== '') return value.trim()
  }
  return ''
}

/** Adapter serving one provider route backed by local models. */
export class LocalModelAdapter extends LlmAdapter {
  /** @type {ModelRegistry} */
  #registry
  /** @type {import('./llama.js').LlamaServers} */
  #servers
  #providerName
  #fetch
  #env
  #defaultMaxTokens
  #active = 0
  #registryOptions

  /**
   * @param {{
   *   models: ReadonlyArray<Record<string, any>>,
   *   providerName: string,
   *   servers: import('./llama.js').LlamaServers,
   *   logger?: any,
   *   defaultMaxTokens?: number,
   *   autoContextCeiling?: number,
   *   fetch?: typeof fetch,
   *   env?: NodeJS.ProcessEnv,
   * }} options - provider facts and injectable I/O.
   */
  constructor(options) {
    super()
    this.#registry = new ModelRegistry(options.models, {
      logger: options.logger,
      autoContextCeiling: options.autoContextCeiling,
    })
    this.#servers = options.servers
    this.#providerName = options.providerName
    this.#fetch = options.fetch ?? globalThis.fetch
    this.#env = options.env ?? process.env
    this.#defaultMaxTokens = options.defaultMaxTokens ?? 4096
    this.#registryOptions = { logger: options.logger, autoContextCeiling: options.autoContextCeiling }
  }

  get active() { return this.#active }

  replaceModels(models, servers) {
    if (this.#active) throw new LlmError('模型正在使用，请结束当前对话请求后保存设置', 'MODEL_BUSY')
    this.#registry = new ModelRegistry(models, this.#registryOptions)
    this.#servers = servers
  }

  /** @returns {ModelRegistry} the catalog this adapter serves. */
  get registry() {
    return this.#registry
  }

  /**
   * Describe the route for the model picker.
   * @param {string} provider - the registered route id.
   * @returns {{ id: string, name: string }} display metadata.
   */
  providerInfo(provider) {
    return { id: provider, name: this.#providerName }
  }

  /**
   * Advertise every configured model.
   * @param {string} provider - the route id.
   * @returns {Promise<Array<Record<string, any>>>} catalog entries.
   */
  async listModels(provider) {
    return this.#registry.list().map((model) => ({
      provider,
      id: model.id,
      name: model.name,
      description: model.backend === 'llamacpp' ? `GGUF · ${model.file}` : `${model.backend === 'ollama' ? 'Ollama' : 'OpenAI-compatible'} · ${model.baseURL}`,
      inputModalities: ['text'],
    }))
  }

  /**
   * Resolve one exact model's metadata without spawning anything.
   * @param {string} provider - the route id.
   * @param {string} model - the requested model id.
   * @param {AbortSignal} [signal] - caller cancellation.
   * @returns {Promise<Record<string, any>>} exact-model metadata.
   */
  async resolveModel(provider, model, signal) {
    const def = this.#require(provider, model)
    if (signal?.aborted) throw abortedFailure(`local-model/${def.id}`)
    const contextWindow = await this.#registry.contextWindow(def)
    return {
      provider,
      id: def.id,
      name: def.name,
      inputModalities: ['text'],
      ...(contextWindow === undefined ? {} : { context: { contextWindow } }),
      ...(def.maxTokens === undefined ? {} : { defaultMaxTokens: def.maxTokens }),
    }
  }

  /**
   * Stream one model call.
   * @param {any} options - the fully assembled request.
   * @returns {AsyncGenerator<Record<string, any>, void, unknown>} harness chunks.
   */
  async *stream(options) {
    this.#active += 1
    try {
      yield* this.#stream(options)
    } finally {
      this.#active -= 1
    }
  }

  async *#stream(options) {
    const def = this.#require(options.provider, options.model)
    const timeout = AbortSignal.timeout(def.requestTimeoutMs ?? 300_000)
    const signal = options.signal ? AbortSignal.any([options.signal, timeout]) : timeout
    const failure = (error) => options.signal?.aborted
      ? abortedFailure(where)
      : timeout.aborted ? new LlmError(`${where} request timed out`, 'TIMEOUT') : error
    const where = `local-model/${def.id}`
    if (signal?.aborted) throw abortedFailure(where)

    let lease
    if (def.backend === 'llamacpp') {
      const ctxSize = await this.#registry.serverContextSize(def)
      try {
        lease = await this.#servers.acquire(def, ctxSize, { signal })
      } catch (error) { throw failure(error) }
    } else {
      lease = { baseURL: def.baseURL, release: () => {} }
    }

    try {
      const nativeOllama = def.backend === 'ollama'
      const url = nativeOllama ? `${ollamaRoot(lease.baseURL)}/api/chat` : chatCompletionsUrl(lease.baseURL)
      const body = (nativeOllama ? buildOllamaRequest : buildChatRequest)(options, def, { defaultMaxTokens: this.#defaultMaxTokens })
      /** @type {Record<string, string>} */
      const headers = {
        ...attributionHeaders(),
        'content-type': 'application/json',
        accept: nativeOllama ? 'application/x-ndjson' : 'text/event-stream',
      }
      const apiKey = resolveApiKey(def, this.#env)
      if (apiKey !== '') headers.authorization = `Bearer ${apiKey}`

      let response
      try {
        response = await this.#fetch(url, {
          method: 'POST',
          headers,
          body: JSON.stringify(body),
          signal,
          redirect: 'error',
        })
      } catch (error) {
        throw failure(transportFailure(error, url))
      }
      if (!response.ok) {
        const text = await response.text().catch(() => '')
        throw httpFailure(response.status, text, url)
      }
      if (response.body === null) {
        throw new LlmError(`${where} returned a response with no body`, 'TRANSPORT')
      }

      /** @type {{ sawDone?: boolean }} */
      const state = {}
      try {
        const events = nativeOllama ? readOllama(response.body, signal, state) : readSse(response.body, signal, state)
        yield* translateChatCompletion(events, { where, state })
      } catch (error) {
        throw failure(error)
      }
    } catch (error) {
      throw failure(error)
    } finally {
      lease.release()
    }
  }

  /**
   * Look one model up, or fail with a route-specific diagnostic.
   * @param {string} provider - the route id from the request.
   * @param {string} model - the model id from the request.
   * @returns {Record<string, any>} the normalized def.
   */
  #require(provider, model) {
    const def = this.#registry.get(model)
    if (def === undefined) {
      const known = this.#registry.list().map((entry) => entry.id).join(', ') || '(none)'
      throw new LlmError(
        `local-model: route "${provider}" has no model "${model}"; configured models: ${known}`,
        'UNKNOWN_MODEL',
      )
    }
    return def
  }
}

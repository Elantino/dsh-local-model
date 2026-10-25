/**
 * The model registry: one place that knows what a configured model is, what
 * context window it can honestly advertise, and what the spawned server should
 * be launched with.
 *
 * GGUF metadata is read once per model and memoized. Metadata is optional: a
 * file that is unreadable, not GGUF, or still being downloaded must not break
 * the provider, so a failed read is logged once and the model keeps working
 * with the configured (or default) context window.
 *
 * @module dsh-local-model/models
 */

import { readGgufMetadata } from './gguf.js'
import { AUTO_CONTEXT_DEFAULT } from './config.js'

/** Registry of the configured models of one provider route. */
export class ModelRegistry {
  /** @type {Map<string, Record<string, any>>} */
  #byId = new Map()
  /** @type {Map<string, Promise<Record<string, any> | undefined>>} */
  #metadata = new Map()
  /** @type {Set<string>} */
  #warned = new Set()
  /** @type {any} */
  #logger
  /** @type {number} */
  #ceiling

  /**
   * @param {ReadonlyArray<Record<string, any>>} models - normalized model defs.
   * @param {{ logger?: any, autoContextCeiling?: number }} [options] - diagnostics and bound.
   */
  constructor(models, options = {}) {
    for (const model of models) this.#byId.set(model.id, model)
    this.#logger = options.logger
    this.#ceiling = typeof options.autoContextCeiling === 'number' && options.autoContextCeiling > 0
      ? options.autoContextCeiling
      : AUTO_CONTEXT_DEFAULT
  }

  /** @returns {number} how many models this provider advertises. */
  get size() {
    return this.#byId.size
  }

  /** @returns {ReadonlyArray<Record<string, any>>} every model, in declaration order. */
  list() {
    return [...this.#byId.values()]
  }

  /**
   * Look one model up by its route id.
   * @param {string} id - model id from the request.
   * @returns {Record<string, any> | undefined} the matching def.
   */
  get(id) {
    return this.#byId.get(id)
  }

  /**
   * Read (and memoize) one model's GGUF metadata.
   * @param {Record<string, any>} model - the normalized def.
   * @returns {Promise<Record<string, any> | undefined>} metadata, or undefined when unavailable.
   */
  async describe(model) {
    if (model.backend !== 'llamacpp') return undefined
    let pending = this.#metadata.get(model.id)
    if (pending === undefined) {
      pending = readGgufMetadata(model.file).catch((error) => {
        this.#warnOnce(
          model.id,
          `local-model: could not read GGUF metadata from "${model.file}" (${error instanceof Error ? error.message : String(error)}); `
            + 'falling back to the configured context window',
        )
        return undefined
      })
      this.#metadata.set(model.id, pending)
    }
    return pending
  }

  /**
   * Resolve the context window this route advertises, in tokens.
   * @param {Record<string, any>} model - the normalized def.
   * @returns {Promise<number | undefined>} the window, or undefined when unknown.
   */
  async contextWindow(model) {
    if (model.backend !== 'llamacpp') {
      return typeof model.contextWindow === 'number' ? model.contextWindow : undefined
    }
    if (typeof model.ctxSize === 'number') return model.ctxSize
    const metadata = await this.describe(model)
    const trained = typeof metadata?.contextLength === 'number' ? metadata.contextLength : undefined
    return trained === undefined ? this.#ceiling : Math.min(trained, this.#ceiling)
  }

  /**
   * Resolve the `--ctx-size` a spawned server should use.
   * @param {Record<string, any>} model - the normalized def.
   * @returns {Promise<number>} a positive token count.
   */
  async serverContextSize(model) {
    return (await this.contextWindow(model)) ?? AUTO_CONTEXT_DEFAULT
  }

  /**
   * Log one diagnostic per model, never once per request.
   * @param {string} modelId - the model the message is about.
   * @param {string} message - the diagnostic.
   */
  #warnOnce(modelId, message) {
    if (this.#warned.has(modelId)) return
    this.#warned.add(modelId)
    this.#logger?.warn?.(message)
  }
}

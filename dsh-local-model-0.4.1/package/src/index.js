/**
 * Plugin entry: `dsh-local-model`.
 *
 * Mount one provider route backed by local model files. The route is
 * registered unconditionally (with whatever models the config declares), so
 * the model picker reflects exactly one source of truth — the profile patch —
 * and nothing has to be "activated" in the UI first.
 *
 * @module dsh-local-model
 */

import { Config, normalizeModels } from './config.js'
import { LocalModelAdapter } from './adapter.js'
import { LlamaServers } from './llama.js'
import { ModelSettings, settingsFile, createSettingsApi } from './settings.js'

export { Config, normalizeModels }

/** Cordis plugin name, shown in loader diagnostics. */
export const name = 'local-model'

/** The adapter registry is the one service this plugin needs. */
export const inject = ['llm']

/**
 * Mount the provider route.
 * @param {import('@deepseek-ai/cordis').Context} ctx - plugin context with `llm` injected.
 * @param {any} config - config already validated by {@link Config}.
 */
export function apply(ctx, config) {
  const store = typeof ctx.inject === 'function' ? new ModelSettings(settingsFile(config), config) : undefined
  const models = normalizeModels({ ...config, models: store?.models ?? config.models })

  const createServers = () => new LlamaServers({
    logger: ctx.logger,
    basePort: config.basePort,
    startupTimeoutMs: config.startupTimeoutMs,
    idleTimeoutMs: config.idleTimeoutMs,
  })
  let servers = createServers()
  let disposed = false
  const retired = new Set()

  const adapter = new LocalModelAdapter({
    models,
    providerName: config.providerName,
    servers,
    logger: ctx.logger,
    defaultMaxTokens: config.defaultMaxTokens,
    autoContextCeiling: config.autoContextCeiling,
  })

  // Spawned servers outlive a request, so they must be released with the plugin
  // fiber; otherwise a profile reload would leak one process per model.
  ctx.effect(() => async () => {
    disposed = true
    await Promise.allSettled([servers.dispose(), ...retired])
  }, 'local-model/servers')

  const registration = ctx.llm.registerAdapter([config.provider.trim()], adapter)

  if (store) ctx.inject(['webServer', 'connection'], (scoped) => {
    const api = createSettingsApi({
      store,
      connection: () => scoped.connection,
      running: () => servers.running,
      save: (models, revision) => {
        if (disposed) throw new Error('插件正在关闭，请稍后重试')
        if (adapter.active) {
          const error = new Error('模型正在使用，请等待当前生成结束后保存')
          error.status = 409
          throw error
        }
        const validated = store.validate(store.mergeSecrets(models))
        const normalized = normalizeModels({ ...config, models: validated })
        // Synchronous persistence + swap: concurrent requests cannot interleave.
        store.save(models, revision)
        const previous = servers
        servers = createServers()
        adapter.replaceModels(normalized, servers)
        const stopping = previous.dispose()
        retired.add(stopping)
        stopping.finally(() => retired.delete(stopping)).catch(() => {})
        registration.replace?.([config.provider.trim()])
      },
    })
    scoped.effect(() => scoped.webServer.register({ kind: 'prefix', path: '/api/local-model', handler: api }), 'local-model/settings-api')
  })

  if (models.length === 0) {
    ctx.logger?.warn?.(
      'local-model: no models configured, so the provider has nothing to offer. '
        + 'Open Settings → 本地模型与 API, or add a `models` entry to your profile cordis.patch.yml '
        + '(see the Configuration section of the plugin README).',
    )
  }
}

/**
 * Plugin configuration: the schema DSH validates and the normalization step
 * that turns user-facing config into the defs the adapter and server manager
 * use.
 *
 * Two design choices matter here:
 *
 * - **Explicit over magical.** A model is one entry in `models`. Nothing is
 *   discovered by scanning directories, so what the model picker shows is
 *   exactly what the file says.
 * - **Fail loud on structure, tolerant on files.** Structural mistakes (a
 *   missing `id`, a duplicate route, a `llamacpp` entry without a file) throw
 *   at mount time, where DSH surfaces them as a plugin failure. A model file
 *   that is missing at runtime only fails that model's request, so a network
 *   drive that is not mounted cannot take the whole provider down.
 *
 * @module dsh-local-model/config
 */

import z from '@deepseek-ai/schemastery'

/** Backends this plugin can drive. */
export const BACKENDS = ['llamacpp', 'openai', 'ollama']

/** Context size used when neither config nor GGUF metadata supplies one. */
export const AUTO_CONTEXT_DEFAULT = 8192

/** Loopback-first default port for spawned llama-server processes. */
export const DEFAULT_BASE_PORT = 8177

/** One model entry. */
const ModelConfig = z.object({
  /** Route-local model id; this is what the DSH model picker submits. */
  id: z.string().required(),
  /** Display name shown in the picker. Defaults to the file name or the id. */
  name: z.string(),
  /**
   * `llamacpp` spawns and owns a `llama-server` process for `file`;
   * `openai` talks to an already-running OpenAI-compatible server.
   * Defaults to `openai` when `baseURL` is set, otherwise `llamacpp`.
   */
  backend: z.union(BACKENDS),
  service: z.union(['llama', 'api']),
  /** Path to a `.gguf` model file. Required by the `llamacpp` backend. */
  file: z.string(),
  /**
   * Force one compute device for `--device`. Empty picks the discrete GPU that
   * `llama-server --list-devices` reports (Vulkan often lists the iGPU first).
   */
  device: z.string(),
  /**
   * Optional vision projector (`mmproj-*.gguf`). Passing it to
   * `llama-server --mmproj` is what turns a text model into a vision model;
   * it is not a model on its own.
   */
  mmproj: z.string(),
  llamaServer: z.string(),
  /**
   * OpenAI-compatible root including the version segment, for example
   * `http://127.0.0.1:11434/v1`. Required by the `openai` backend.
   */
  baseURL: z.string(),
  /** Bearer token for the `openai` backend. Prefer `apiKeyEnv` over inline secrets. */
  apiKey: z.string(),
  /** Environment variable holding the bearer token for the `openai` backend. */
  apiKeyEnv: z.string(),
  /** Model name the endpoint expects, when it differs from `id`. */
  remoteModel: z.string(),
  /** Advertised context window for the `openai` backend, in tokens. */
  contextWindow: z.number().step(1).min(1),
  /** Output cap used when a request carries none. */
  maxTokens: z.number().step(1).min(1),
  requestTimeoutMs: z.number().step(1).min(1000),
  /** `llama-server --ctx-size`. Also becomes the advertised context window. */
  ctxSize: z.number().step(1).min(1),
  /** `llama-server --n-gpu-layers`. 0 keeps every layer on the CPU. */
  gpuLayers: z.number().step(1).min(0),
  /** `llama-server --threads`. */
  threads: z.number().step(1).min(1),
  /** `llama-server --batch-size`. */
  batchSize: z.number().step(1).min(1),
  /** `llama-server --parallel` slots. */
  parallel: z.number().step(1).min(1),
  /** `llama-server --jinja`, required for native tool calling. */
  jinja: z.boolean(),
  /** `-fa on|off|auto`. Auto passes nothing and lets llama.cpp decide. */
  flashAttention: z.union(['auto', 'on', 'off']),
  /**
   * `-ctk` / `-ctv` KV-cache precision. `q8_0` roughly halves the KV cache and
   * is what makes a 32K context fit an 8 GB card with a 9B Q4 model; it
   * requires Flash Attention, which the launcher turns on automatically.
   */
  kvCacheType: z.union(['f16', 'q8_0', 'q4_0']),
  /**
   * Let a thinking model emit its reasoning block. `false` suppresses it for
   * Ollama (`think: false`) and for a managed llama-server
   * (`chat_template_kwargs.enable_thinking: false`); `true` leaves the model's
   * own default in place.
   */
  think: z.boolean(),
  /** `llama-server --chat-template` file. */
  chatTemplate: z.string(),
  /** Extra `llama-server` flags, appended verbatim. */
  extraArgs: z.array(z.string()),
  /**
   * Extra JSON fields merged into the chat-completions body — the escape hatch
   * for server-specific sampling knobs such as
   * `{"chat_template_kwargs": {"enable_thinking": false}}`.
   */
  extraBody: z.any(),
})

/** The plugin's configuration schema. */
export const Config = z.object({
  /** Provider route id registered with `ctx.llm`. */
  provider: z.string().default('local'),
  /** Display name of the provider group in the model picker. */
  providerName: z.string().default('Local Models'),
  stateFile: z.string(),
  /** Path to `llama-server`. Empty means search `LLAMA_SERVER` and `PATH`. */
  llamaServer: z.string().default(''),
  /**
   * Engine build to download when the plugin provisions one: `auto` follows the
   * detected GPU (CUDA on NVIDIA, Vulkan elsewhere), `cpu` only as a last resort.
   */
  engineVariant: z.union(['auto', 'cuda', 'vulkan', 'cpu']).default('auto'),
  /** First port a spawned server may use; later models take the next free port. */
  basePort: z.number().step(1).min(1024).max(65535).default(DEFAULT_BASE_PORT),
  /** How long to wait for a spawned server to report healthy. */
  startupTimeoutMs: z.number().step(1).min(1000).default(180_000),
  /** Stop a spawned server after this long with no requests. 0 keeps it loaded. */
  idleTimeoutMs: z.number().step(1).min(0).default(600_000),
  /** Output cap used when neither the request nor the model entry sets one. */
  defaultMaxTokens: z.number().step(1).min(1).default(4096),
  requestTimeoutMs: z.number().step(1).min(1000).default(300_000),
  /** Upper bound for a context window read from GGUF metadata. */
  autoContextCeiling: z.number().step(1).min(1024).default(AUTO_CONTEXT_DEFAULT),
  /** The models this provider offers. */
  models: z.array(ModelConfig).default([]),
})

/**
 * Turn validated config into normalized model definitions.
 *
 * @param {any} config - config already validated by {@link Config}.
 * @returns {ReadonlyArray<Record<string, any>>} one normalized def per model,
 *   in declaration order, with defaults applied.
 * @throws {Error} describing every structural problem at once.
 */
export function normalizeModels(config) {
  /** @type {string[]} */
  const issues = []
  const provider = typeof config.provider === 'string' ? config.provider.trim() : ''
  if (provider === '') issues.push('`provider` must be a non-empty route id')
  else if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(provider)) {
    issues.push(`\`provider\` "${provider}" may only contain letters, digits, ".", "_" and "-"`)
  }

  const entries = Array.isArray(config.models) ? config.models : []
  const seen = new Set()
  /** @type {Array<Record<string, any>>} */
  const models = []

  entries.forEach((entry, index) => {
    const where = `models[${index}]`
    const id = typeof entry?.id === 'string' ? entry.id.trim() : ''
    if (id === '') {
      issues.push(`${where}.id is required`)
      return
    }
    if (seen.has(id)) {
      issues.push(`${where}.id "${id}" is declared more than once`)
      return
    }
    seen.add(id)

    const backend = resolveBackend(entry, id, where, issues)
    const file = trimmed(entry.file)
    const baseURL = trimmed(entry.baseURL)
    if (backend === 'llamacpp' && file === '') {
      issues.push(`${where} ("${id}") uses backend "llamacpp" but has no \`file\`; point it at a .gguf path`)
    }
    if (backend !== 'llamacpp') {
      if (baseURL === '') {
        issues.push(`${where} ("${id}") uses backend "${backend}" but has no \`baseURL\` (for example http://127.0.0.1:11434/v1)`)
      } else if (!/^https?:\/\//i.test(baseURL)) {
        issues.push(`${where} ("${id}") has a \`baseURL\` that must start with http:// or https://`)
      } else {
        try {
          const url = new URL(baseURL)
          if (url.username || url.password || url.search || url.hash) throw new Error()
        } catch {
          issues.push(`${where} baseURL must be a valid URL without embedded credentials, query or fragment`)
        }
      }
    }
    const extraArgs = Array.isArray(entry.extraArgs) ? entry.extraArgs.map(String) : []
    for (const arg of extraArgs) {
      if (/^(?:--(?:host|port|model|model-url|hf-repo|hf-file|ctx-size|parallel|api-prefix|ssl-key-file|ssl-cert-file|api-key|api-key-file|chat-template|chat-template-file)|-[mc]|-np|-hf)(?:=|$)/.test(arg)) {
        issues.push(`${where}.extraArgs cannot override managed argument ${arg}`)
      }
    }
    if (file !== '' && !/\.gguf$/i.test(file)) {
      issues.push(`${where} ("${id}") \`file\` must end in .gguf (got "${file}")`)
    }
    const mmproj = trimmed(entry.mmproj)
    if (mmproj !== '') {
      if (backend === 'ollama') {
        issues.push(`${where} ("${id}") Ollama 后端不需要 \`mmproj\`；视觉模型直接写 Ollama 里的模型名`)
      } else if (!/\.gguf$/i.test(mmproj)) {
        issues.push(`${where} ("${id}") \`mmproj\` must end in .gguf (got "${mmproj}")`)
      }
    }

    const name = trimmed(entry.name) || displayNameOf(file, id)
    const window = positive(backend === 'llamacpp' ? entry.ctxSize : entry.contextWindow)
    if (window !== undefined && positive(entry.maxTokens) >= window) {
      issues.push(`${where}.maxTokens 必须小于上下文长度，为输入消息保留空间`)
    }
    models.push({
      id,
      name,
      backend,
      // llama.cpp runtime
      file,
      device: trimmed(entry.device),
      mmproj,
      llamaServer: trimmed(entry.llamaServer) || trimmed(config.llamaServer),
      ctxSize: positive(entry.ctxSize),
      gpuLayers: entry.gpuLayers === undefined || entry.gpuLayers === null ? 999 : entry.gpuLayers,
      threads: positive(entry.threads),
      batchSize: positive(entry.batchSize),
      parallel: positive(entry.parallel) ?? 1,
      jinja: entry.jinja !== false,
      flashAttention: trimmed(entry.flashAttention) || 'auto',
      kvCacheType: trimmed(entry.kvCacheType) || 'f16',
      think: entry.think !== false,
      chatTemplate: trimmed(entry.chatTemplate),
      extraArgs,
      // OpenAI-compatible endpoint
      baseURL,
      // `llama` marks an existing llama.cpp server: it understands the same
      // Jinja template kwargs as a managed one, so the thinking switch applies.
      service: backend === 'openai' ? (trimmed(entry.service) || 'api') : '',
      apiKey: trimmed(entry.apiKey),
      apiKeyEnv: trimmed(entry.apiKeyEnv),
      remoteModel: trimmed(entry.remoteModel) || id,
      contextWindow: positive(entry.contextWindow),
      maxTokens: positive(entry.maxTokens),
      requestTimeoutMs: positive(entry.requestTimeoutMs) ?? positive(config.requestTimeoutMs) ?? 300_000,
      // request escape hatch
      extraBody: isPlainObject(entry.extraBody) ? entry.extraBody : {},
    })
  })

  if (issues.length > 0) {
    throw new Error(`local-model: invalid configuration:\n  - ${issues.join('\n  - ')}`)
  }
  return Object.freeze(models)
}

/**
 * Resolve the backend for one entry: explicit value, else inferred from `baseURL`.
 * @param {Record<string, unknown>} entry - raw model entry.
 * @param {string} id - its model id (for diagnostics).
 * @param {string} where - its config path (for diagnostics).
 * @param {string[]} issues - collected problems.
 * @returns {'llamacpp' | 'openai'} the resolved backend.
 */
function resolveBackend(entry, id, where, issues) {
  const declared = trimmed(entry.backend)
  if (declared !== '') {
    if (!BACKENDS.includes(declared)) {
      issues.push(`${where} ("${id}") has backend "${declared}"; expected one of ${BACKENDS.join(', ')}`)
      return 'llamacpp'
    }
    return /** @type {'llamacpp' | 'openai'} */ (declared)
  }
  return trimmed(entry.baseURL) !== '' ? 'openai' : 'llamacpp'
}

/**
 * Normalize one optional string field.
 * @param {unknown} value - raw value.
 * @returns {string} trimmed string, or an empty string.
 */
function trimmed(value) {
  return typeof value === 'string' ? value.trim() : ''
}

/**
 * Normalize one optional positive integer field.
 * @param {unknown} value - raw value.
 * @returns {number | undefined} the value, when positive.
 */
function positive(value) {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.floor(value) : undefined
}

/**
 * Derive a picker label from the model file, then the id.
 * @param {string} file - the `.gguf` path, if any.
 * @param {string} id - the model id.
 * @returns {string} a human-readable label.
 */
function displayNameOf(file, id) {
  if (file === '') return id
  const base = file.split(/[\\/]/).pop() ?? id
  return base.replace(/\.gguf$/i, '')
}

/**
 * Narrow a value to a plain object.
 * @param {unknown} value - candidate.
 * @returns {value is Record<string, unknown>} true for plain objects.
 */
function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

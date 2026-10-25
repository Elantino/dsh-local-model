/*
 * DSH ModuleLoader bundle for the browser half. React comes from the shell, so
 * there is no build step; every colour is a theme variable so the page follows
 * the active scheme.
 *
 * The page is one settings section. It edits the plugin's own JSON store over
 * `/api/local-model/*` (same-origin, admitted by the Host connection service),
 * never the Cordis patch: the profile's YAML stays a fallback for headless use.
 */
window.__ModuleLoader__.load({
  id: 'dsh-local-model',
  factory: (require) => {
    const React = require('react')
    const { createElement: h, useState, useEffect } = React
    const base = '/api/local-model'
    const panel = { border: '1px solid var(--border-color, #8885)', borderRadius: 10, padding: 16, marginBottom: 16 }
    const inputStyle = { width: '100%', boxSizing: 'border-box', padding: '8px 10px', borderRadius: 6, border: '1px solid var(--border-color, #8887)', background: 'var(--background-color, transparent)', color: 'inherit' }
    const buttonStyle = { padding: '7px 12px', borderRadius: 6, border: '1px solid var(--border-color, #8887)', background: 'var(--background-color, transparent)', color: 'inherit', cursor: 'pointer' }
    const row = { display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }
    const grid = { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 250px), 1fr))', gap: 14 }
    const muted = { fontSize: 12, opacity: 0.75, lineHeight: 1.7 }
    const badge = { display: 'inline-block', padding: '2px 8px', marginRight: 6, borderRadius: 999, border: '1px solid var(--border-color, #8887)', fontSize: 11 }
    const CONTEXT_PRESETS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    const CAPABILITY_LABELS = { tools: '工具调用', thinking: '思考', vision: '视觉', completion: '对话', embedding: '向量' }

    async function api(route, body, signal) {
      const response = await fetch(base + route, {
        method: body === undefined ? 'GET' : 'POST', credentials: 'same-origin',
        headers: body === undefined ? {} : { 'content-type': 'application/json' },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
        signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(30_000)]) : AbortSignal.timeout(30_000),
      })
      const value = await response.json()
      if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`)
      return value
    }

    function fresh() {
      // 2048 leaves room for a thinking model to finish its reasoning block and
      // still answer; the model card suggests more when it needs it.
      return {
        id: '', name: '', backend: 'llamacpp', file: '', ctxSize: 8192, maxTokens: 2048,
        gpuLayers: 999, flashAttention: 'auto', kvCacheType: 'f16', think: true, requestTimeoutMs: 300000,
      }
    }

    function SettingsPage() {
      const [saved, setSaved] = useState(null)
      const [draft, setDraft] = useState(null)
      const [editing, setEditing] = useState(null)
      const [busy, setBusy] = useState(false)
      const [error, setError] = useState('')
      const [notice, setNotice] = useState('')
      const [listing, setListing] = useState(null)
      const [directory, setDirectory] = useState('')
      const [discovered, setDiscovered] = useState([])
      const [metadata, setMetadata] = useState(null)
      const [detected, setDetected] = useState(null)
      const [probe, setProbe] = useState(null)
      const [engine, setEngine] = useState(null)

      async function run(task) {
        setBusy(true); setError(''); setNotice('')
        try { await task() } catch (err) { setError(err.message) } finally { setBusy(false) }
      }
      useEffect(() => {
        const controller = new AbortController()
        api('/settings', undefined, controller.signal).then((value) => {
          setSaved(value)
          if (value.models.length === 0) {
            api('/detect', undefined, controller.signal).then((found) => setDetected(found)).catch(() => {})
          }
        }).catch((err) => {
          if (!controller.signal.aborted) setError(err.message)
        })
        api('/engine', undefined, controller.signal).then(setEngine).catch(() => {})
        return () => controller.abort()
      }, [])

      // Poll while a download is running; the Host returns the live phase.
      const enginePhase = engine?.status?.phase
      useEffect(() => {
        if (enginePhase !== 'metadata' && enginePhase !== 'download' && enginePhase !== 'extract') return undefined
        const timer = setInterval(() => {
          api('/engine').then(setEngine).catch(() => {})
        }, 1500)
        return () => clearInterval(timer)
      }, [enginePhase])

      async function installEngine(variant) {
        await run(async () => {
          setEngine(await api('/engine/install', { variant }))
          setNotice('开始下载推理引擎，完成后 GGUF 模型即可直接启动。')
        })
      }

      async function removeEngine() {
        await run(async () => {
          setEngine(await api('/engine/remove', {}))
          setNotice('已移除下载的推理引擎。')
        })
      }

      const button = (label, onClick, extra = {}) => h('button', { type: 'button', style: buttonStyle, disabled: busy, onClick, ...extra }, label)
      const update = (key, value) => setDraft((current) => ({ ...current, [key]: value }))
      const field = (label, key, { number = false, hint, ...props } = {}) => h('label', { style: { display: 'grid', gap: 6 } },
        h('span', null, label),
        h('input', { style: inputStyle, disabled: busy, type: number ? 'number' : 'text',
          value: draft[key] ?? '', ...(number ? { min: 1, step: 1 } : {}), ...props,
          onChange: (event) => update(key, number ? (event.target.value === '' ? undefined : Number(event.target.value)) : event.target.value) }),
        hint ? h('small', { style: muted }, hint) : null)

      function start(model) {
        setEditing(model?.id ?? null)
        setDraft(model ? { ...model } : fresh())
        setListing(null); setDiscovered([]); setMetadata(null); setProbe(null); setError(''); setNotice('')
      }

      function changeType(type) {
        setDraft((current) => {
          // Clear backend-specific fields, including credentials on type changes.
          const common = { id: current.id, name: current.name, maxTokens: current.maxTokens ?? 1024, think: current.think !== false, requestTimeoutMs: current.requestTimeoutMs ?? 300000 }
          if (type === 'gguf') return { ...common, backend: 'llamacpp', ctxSize: 8192, gpuLayers: 999, flashAttention: 'auto', kvCacheType: 'f16', file: '' }
          if (type === 'ollama') return { ...common, backend: 'ollama', baseURL: 'http://127.0.0.1:11434', remoteModel: '', contextWindow: 8192, apiKey: '' }
          return { ...common, backend: 'openai', service: type, baseURL: type === 'llama' ? 'http://127.0.0.1:8080/v1' : '', remoteModel: '', contextWindow: 8192, apiKey: '' }
        })
        setListing(null); setDiscovered([]); setMetadata(null); setProbe(null)
      }

      async function saveModel() {
        const model = { ...draft, id: draft.id.trim(), name: draft.name?.trim() }
        if (!model.id) throw new Error('请填写模型 ID')
        if (saved.models.some((item) => item.id === model.id && item.id !== editing)) throw new Error('模型 ID 已存在')
        const models = editing ? saved.models.map((item) => item.id === editing ? model : item) : [...saved.models, model]
        const result = await api('/settings', { models, revision: saved.revision })
        setSaved(result); setDraft(null); setListing(null)
        setNotice('已保存并生效，可在对话的模型选择器中选择。')
      }

      async function browse(target) {
        const result = await api(`/browse?directory=${encodeURIComponent(target)}`)
        setListing(result); setDirectory(result.directory)
      }

      async function choose(file) {
        const info = await api('/inspect', { file })
        // A vision build ships the main model and an `mmproj-*.gguf` projector;
        // picking the projector is the common mistake, and llama-server only
        // reports it as an opaque load failure. Stop here with the reason.
        if (info.kind === 'mmproj') throw new Error(info.warning)
        setMetadata(info)
        setDraft((current) => ({ ...current, file,
          id: current.id || file.split(/[\\/]/).pop().replace(/\.gguf$/i, ''),
          // Start at the model's own context, but never silently allocate a
          // 128K KV cache from one click; the user can still raise it.
          ctxSize: info.suggestedContext ?? Math.min(info.contextLength ?? 8192, 32768),
        }))
        setListing(null)
        if (info.llamaServer) {
          setDraft((current) => ({ ...current, llamaServer: current.llamaServer || info.llamaServer }))
          setNotice(`已自动找到 llama-server：${info.llamaServer}`)
        }
      }

      async function inspectOllamaModel() {
        const info = await api('/inspect', { baseURL: draft.baseURL, remoteModel: draft.remoteModel })
        setMetadata({ ...info, kind: 'ollama' })
        if (info.suggestedContext) update(contextKey, info.suggestedContext)
        return info
      }

      async function addDetected(entry) {
        await run(async () => {
          setEditing(null)
          setDraft({
            ...fresh(),
            backend: 'ollama',
            baseURL: detected?.baseURL ?? 'http://127.0.0.1:11434',
            remoteModel: entry.name,
            id: entry.name.replace(/[:/]/g, '-'),
            contextWindow: 8192,
          })
          setProbe(null); setListing(null); setError(''); setNotice('')
          const info = await inspectOllamaModel()
          setNotice(`已读取 ${entry.name}：建议上下文 ${info.suggestedContext ?? '未知'}，确认后保存。`)
        })
      }

      async function testConnection() {
        const result = await api('/test', { model: { ...draft, id: draft.id?.trim() || 'probe' } })
        setProbe(result)
      }

      const type = draft?.backend === 'llamacpp' ? 'gguf' : draft?.backend === 'ollama' ? 'ollama' : draft?.service === 'llama' ? 'llama' : 'api'
      const contextKey = type === 'gguf' ? 'ctxSize' : 'contextWindow'
      const contextValue = draft?.[contextKey]
      const supportsThinking = type === 'gguf' || type === 'ollama'
      return h('div', { style: { padding: 20, maxWidth: 1000, margin: '0 auto', color: 'inherit' } },
        h('h2', null, '本地模型与 API'),
        h('p', { style: muted }, '添加本地 GGUF、已有 llama.cpp / Ollama 服务，或 OpenAI 兼容的模型 API。保存后无需重启。'),
        error ? h('div', { role: 'alert', style: { ...panel, borderColor: '#c44', whiteSpace: 'pre-wrap' } }, error,
          button('刷新设置', () => run(async () => { setSaved(await api('/settings')); setDraft(null); setListing(null) }))) : null,
        notice ? h('div', { role: 'status', style: panel }, notice) : null,
        engine === null ? null : (() => {
          const status = engine.status ?? {}
          const phase = status.phase ?? 'idle'
          const downloading = phase === 'metadata' || phase === 'download' || phase === 'extract'
          const percent = status.total > 0 ? Math.min(100, Math.round((status.received / status.total) * 100)) : 0
          return h('section', { style: panel, 'aria-label': '推理引擎' },
            h('div', { style: { ...row, justifyContent: 'space-between' } },
              h('strong', null, '推理引擎'),
              h('span', { style: muted }, engine.resolved
                ? `已就绪${engine.installed ? ` · 内置 ${engine.installed.tag}` : ' · 使用现有安装'}`
                : phase === 'error' ? '下载失败' : downloading ? '下载中' : '未安装')),
            h('p', { style: { ...muted, overflowWrap: 'anywhere' } }, engine.resolved
              ?? 'GGUF 模型需要 llama-server。下载后插件自己管理，不用另装 llama.cpp，也不用改 PATH。'),
            downloading ? h('div', { style: { margin: '8px 0' } },
              h('div', { style: { height: 6, borderRadius: 3, background: 'var(--border-color, #8884)', overflow: 'hidden' } },
                h('div', { style: { height: '100%', width: `${percent}%`, background: 'var(--accent-color, #6a9)' } })),
              h('small', { style: muted }, phase === 'metadata'
                ? '正在查询最新版本…'
                : phase === 'extract' ? '正在解压…'
                  : `${(status.received / 1024 / 1024).toFixed(1)} / ${(status.total / 1024 / 1024).toFixed(1)} MB`)) : null,
            phase === 'error' && status.error ? h('p', { style: { ...muted, color: '#c44' } }, status.error) : null,
            h('div', { style: row },
              engine.installed
                ? h(React.Fragment, null,
                  button('重新下载（Vulkan）', () => installEngine('vulkan'), { disabled: downloading || busy }),
                  button('移除', () => removeEngine(), { disabled: downloading || busy }))
                : engine.resolved ? null : h(React.Fragment, null,
                  button('下载 Vulkan 引擎（约 35 MB）', () => installEngine('vulkan'), { disabled: downloading || busy }),
                  button('下载 CPU 引擎（约 20 MB）', () => installEngine('cpu'), { disabled: downloading || busy }))))
        })(),
        !saved ? h('p', null, error ? '设置未加载。' : '正在加载设置…') : h(React.Fragment, null,
          h('div', { style: { ...row, marginBottom: 16 } },
            button('添加模型', () => start(null), { disabled: busy || Boolean(draft) }),
            button('刷新', () => run(async () => { setSaved(await api('/settings')) }), { disabled: busy || Boolean(draft) })),
          saved.models.length === 0 && !draft ? (detected?.reachable && detected.models.length
            ? h('section', { style: panel, 'aria-label': '检测到的 Ollama 模型' },
              h('h3', null, '检测到本机 Ollama'),
              h('p', { style: muted }, `${detected.baseURL} · ${detected.models.length} 个模型。点一个开始，模型名和上下文会自动填入。`),
              h('div', { style: row }, ...detected.models.slice(0, 12).map((entry) => button(entry.name, () => addDetected(entry), { disabled: busy }))))
            : h('p', { style: panel }, '尚未添加模型。点击“添加模型”开始。'
              + (detected && !detected.reachable ? ' 也可以先启动 Ollama（默认 127.0.0.1:11434），本页会自动检测。' : ''))) : null,
          ...saved.models.map((model) => h('div', { key: model.id, style: panel },
            h('div', { style: { ...row, justifyContent: 'space-between' } },
              h('strong', null, model.name || model.id),
              h('span', { style: muted }, model.backend === 'llamacpp' ? 'GGUF' : model.backend === 'ollama' ? 'Ollama' : 'OpenAI 兼容 API')),
            h('p', { style: { ...muted, overflowWrap: 'anywhere' } }, `${model.id} · ${model.file || model.baseURL || ''}`),
            h('p', { style: muted }, `上下文：${model.ctxSize ?? model.contextWindow ?? '自动'} tokens · 输出上限：${model.maxTokens ?? '默认'}${model.think === false ? ' · 已关闭思考' : ''}`
              + (saved.running?.includes(model.id) ? ' · 进程运行中' : '')),
            h('div', { style: row }, button('编辑', () => start(model), { disabled: busy || Boolean(draft) }),
              button('删除', () => run(async () => {
                const models = saved.models.filter((item) => item.id !== model.id)
                setSaved(await api('/settings', { models, revision: saved.revision }))
                setNotice('模型已移除。')
              }), { disabled: busy || Boolean(draft) })))),
          draft ? h('form', { style: panel, onSubmit: (event) => { event.preventDefault(); if (!busy) void run(saveModel) } },
            h('h3', null, editing ? '编辑模型' : '添加模型'),
            h('fieldset', { disabled: busy, style: { border: 0, margin: 0, padding: 0, minWidth: 0 } },
              h('label', { style: { display: 'grid', gap: 6, marginBottom: 16 } }, '接入方式',
                h('select', { value: type, style: inputStyle, onChange: (event) => changeType(event.target.value) },
                  h('option', { value: 'gguf' }, '本地 GGUF 文件（自动管理 llama-server）'),
                  h('option', { value: 'llama' }, '已有 llama.cpp / llama-server 服务'),
                  h('option', { value: 'ollama' }, 'Ollama（原生接口，可调整运行上下文）'),
                  h('option', { value: 'api' }, '模型 API（OpenAI 兼容）'))),
              h('div', { style: grid },
                field('模型 ID', 'id', { required: true, disabled: busy || Boolean(editing), hint: '插件内唯一；保存后不能更改。' }),
                field('显示名称', 'name')),
              type === 'gguf' ? h('div', { style: { marginTop: 16 } },
                field('GGUF 文件路径', 'file', { required: true, placeholder: 'D:\\models\\model.gguf' }),
                h('div', { style: { ...row, marginTop: 8 } },
                  button('浏览本机 GGUF 文件', () => run(() => browse(''))),
                  button('读取模型信息', () => run(() => choose(draft.file)))),
                metadata?.kind === 'gguf' ? h('p', { style: muted }, `架构：${metadata.architecture ?? '未知'}；模型声明上下文：${metadata.contextLength ?? '未知'}；元数据${metadata.complete ? '完整' : '部分读取'}`) : null,
                h('div', { style: { ...grid, marginTop: 16 } },
                  field('llama-server 可执行文件（可选）', 'llamaServer', { placeholder: '留空则自动查找 PATH、LLAMA_SERVER 与常见目录' }),
                  field('GPU 层数', 'gpuLayers', { number: true, min: 0, hint: '0 = CPU；999 = 尽可能使用 GPU。' }),
                  h('label', { style: { display: 'grid', gap: 6 } }, 'KV 缓存精度',
                    h('select', { style: inputStyle, disabled: busy, value: draft.kvCacheType ?? 'f16',
                      onChange: (event) => update('kvCacheType', event.target.value) },
                    h('option', { value: 'f16' }, 'f16（默认，精度最高）'),
                    h('option', { value: 'q8_0' }, 'q8_0（省约一半 KV 显存）'),
                    h('option', { value: 'q4_0' }, 'q4_0（最省，精度略降）'))),
                  h('label', { style: { display: 'grid', gap: 6 } }, 'Flash Attention',
                    h('select', { style: inputStyle, disabled: busy, value: draft.flashAttention ?? 'auto',
                      onChange: (event) => update('flashAttention', event.target.value) },
                    h('option', { value: 'auto' }, 'auto（交给 llama.cpp）'),
                    h('option', { value: 'on' }, 'on'),
                    h('option', { value: 'off' }, 'off')))),
                h('p', { style: muted }, '8GB 显存跑 9B Q4：上下文 32K + KV 选 q8_0（Flash Attention 会自动开启）实测可用；'
                  + '上下文越大 KV 越占显存，放不下就降上下文或选 q4_0。')) : h('div', { style: { ...grid, marginTop: 16 } },
                field('服务地址', 'baseURL', { required: true, placeholder: type === 'ollama' ? 'http://127.0.0.1:11434' : 'https://api.example.com/v1' }),
                field('服务端模型名', 'remoteModel', { required: true, placeholder: type === 'ollama' ? 'qwen3:8b' : '服务商要求的模型 ID' }),
                field('API Key（可选）', 'apiKey', { type: 'password', autoComplete: 'new-password', placeholder: draft.apiKeyConfigured ? '已保存；不填写则保留原密钥' : '输入 API Key', hint: '密钥仅存于 DSH 主机；不会随设置读取返回。' }),
                field('密钥环境变量名（可选）', 'apiKeyEnv', { placeholder: '例如 MODEL_API_KEY' }),
                h('div', { style: row }, button('清除保存的密钥', () => { update('apiKey', ''); update('apiKeyConfigured', false) }),
                  button('读取服务端模型列表', () => run(async () => {
                    const result = await api('/discover', { model: { ...draft, id: draft.id.trim() || 'discovery' } })
                    setDiscovered(result.models)
                    setNotice(result.models.length ? '已读取模型列表，请选择模型。' : '服务未列出模型，可手动填写模型名。')
                  })),
                  type === 'ollama' ? button('读取模型能力', () => run(inspectOllamaModel)) : null),
                discovered.length ? h('label', { style: { display: 'grid', gap: 6 } }, '选择服务端模型',
                  h('select', { style: inputStyle, value: draft.remoteModel ?? '', onChange: (event) => {
                    update('remoteModel', event.target.value)
                    if (!draft.id) update('id', event.target.value)
                  } }, h('option', { value: '' }, '请选择'),
                  ...discovered.map((model) => h('option', { key: model.id, value: model.id }, model.id)))) : null,
                metadata?.kind === 'ollama' ? h('div', { style: { marginTop: 10 } },
                  h('div', { style: muted }, (metadata.capabilities ?? []).map((capability) => h('span', { key: capability, style: badge }, CAPABILITY_LABELS[capability] ?? capability)),
                    `模型声明上下文 ${metadata.trainedContext ?? '未知'}`,
                    metadata.configuredContext ? `；Modelfile 固定 ${metadata.configuredContext}` : '',
                    metadata.suggestedContext ? `；建议 ${metadata.suggestedContext}` : ''),
                  h('p', { style: muted }, (metadata.capabilities ?? []).includes('thinking') && draft.think !== false
                    ? '该模型支持思考：思考内容也占用“最大输出 tokens”，输出上限太小会把回答挤掉。'
                    : '该模型未声明思考能力。')) : null),
              h('div', { style: { ...grid, marginTop: 16 } },
                h('label', { style: { display: 'grid', gap: 6 } }, '上下文预设',
                  h('select', { style: inputStyle, value: CONTEXT_PRESETS.includes(contextValue) ? contextValue : '',
                    onChange: (event) => { if (event.target.value) update(contextKey, Number(event.target.value)) } },
                  h('option', { value: '' }, '自定义'),
                  ...CONTEXT_PRESETS.map((n) => h('option', { key: n, value: n }, `${n / 1024}K`)))),
                field('上下文长度（tokens）', contextKey, { number: true, required: true }),
                field('最大输出 tokens', 'maxTokens', { number: true, required: true, hint: '思考模型建议至少 2048：思考内容也占用这个额度。' }),
                field('单次请求超时（毫秒）', 'requestTimeoutMs', { number: true, min: 1000, required: true })),
              supportsThinking ? h('label', { style: { ...row, marginTop: 8 } },
                h('input', { type: 'checkbox', checked: draft.think !== false, disabled: busy,
                  onChange: (event) => update('think', event.target.checked) }),
                h('span', null, '启用思考（thinking）'),
                h('small', { style: muted }, type === 'ollama' ? '关闭后通过 think:false 发送，回答更快更省 token。' : '关闭后通过 chat_template_kwargs 发送 enable_thinking:false。')) : null,
              contextValue > 32768 ? h('p', { style: { ...muted, color: '#c84' } }, '上下文超过 32K：KV cache 会明显占用内存 / 显存，加载也更慢。') : null,
              h('p', { style: muted }, type === 'gguf'
                ? '此长度用于 llama-server 每个并发槽位。上下文越大，内存和显存占用越高；不要超过模型支持范围。'
                : type === 'ollama' ? '上下文通过原生接口 options.num_ctx 发送给 Ollama，同时用于 DSH 的上下文管理。修改后 Ollama 会在下一次请求时重新加载模型。'
                  : '此长度告诉 DSH 何时裁剪历史；已有 llama.cpp 服务的启动上下文、云端 API 的硬上限需在服务端配置。'),
              h('div', { style: row }, h('button', { type: 'submit', style: buttonStyle, disabled: busy }, busy ? '处理中…' : '保存并生效'),
                button('测试连接', () => run(testConnection)),
                button('取消', () => { setDraft(null); setListing(null); setProbe(null); setError('') })),
              probe ? h('p', { role: 'status', style: muted }, `${probe.ok ? '✓' : '✗'} ${probe.detail}`) : null)) : null,
          listing ? h('section', { style: panel, 'aria-label': '选择 GGUF 文件' },
            h('h3', null, '选择 DSH 所在电脑上的 GGUF 文件'),
            h('p', { style: muted }, '直接引用文件路径，不上传模型。远程访问 DSH 时，这里显示的是 DSH 服务器磁盘。'),
            h('div', { style: row },
              h('input', { style: { ...inputStyle, flex: 1, minWidth: 160 }, disabled: busy, value: directory, placeholder: '输入绝对目录路径', onChange: (event) => setDirectory(event.target.value) }),
              button('打开目录', () => run(() => browse(directory))),
              button('上级目录', () => run(() => browse(listing.parent ?? ''))),
              button('关闭', () => setListing(null))),
            h('div', { style: { maxHeight: 360, overflowY: 'auto', marginTop: 12 } },
              listing.entries.length ? listing.entries.map((entry) => h('button', {
                key: entry.path, type: 'button', disabled: busy,
                style: { ...buttonStyle, display: 'block', width: '100%', textAlign: 'left', marginBottom: 4, overflowWrap: 'anywhere' },
                onClick: () => run(() => entry.directory ? browse(entry.path) : choose(entry.path)),
              }, `${entry.directory ? '目录' : /^mmproj[-_.]/i.test(entry.name) ? '视觉投影（不能单独用）' : 'GGUF'} · ${entry.name}`)) : h('p', null, '此目录下没有子目录或 GGUF 文件。')),
            listing.truncated ? h('p', { style: muted }, '仅显示前 500 项，请输入更具体的目录路径。') : null) : null,
          h('p', { style: muted }, '设置保存在 DSH_HOME/local-model 下；可用 stateFile 配置隔离不同 profile。请求生成期间不可保存。API 模式仅支持 OpenAI Chat Completions 兼容协议。')))
    }

    return {
      name: 'local-model', inject: ['slots'],
      apply(ctx) {
        const mount = () => ctx.slots.inject('settings.section', () => ctx.slots.register({
          name: 'settings.section', id: 'local-model', order: 36,
          label: () => '本地模型与 API',
        }, () => h(SettingsPage)))
        // A client plugin unload must withdraw its section; `effect` is where
        // the shell expects that ownership.
        if (typeof ctx.effect === 'function') ctx.effect(mount, 'local-model: settings section')
        else mount()
      },
      __test: { SettingsPage, fresh },
    }
  },
})

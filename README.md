# dsh-local-model
让 DSH 直接使用本地 GGUF 模型文件，或接入已有的 llama.cpp / Ollama / OpenAI 兼容 API；支持设置上下文长度、关闭思考、自动下载推理引擎。
# dsh-local-model 用户指南

> 适用版本：**0.4.x**（DSH `0.2.x`，Node.js ≥ 22.19）

一句话介绍：

> 让 DSH 直接使用本地 GGUF 模型文件，或接入已有的 llama.cpp / Ollama / OpenAI 兼容 API；在 **设置 → 本地模型与 API** 里点选即可，支持设置上下文长度、关闭思考、自动下载推理引擎。

---

## 本文包含

- [它能解决什么](#what)
- [安装](#install)
- [四种接入方式](#modes)
- [第一次使用](#first-run)
- [设置页逐项说明](#settings)
- [在对话里怎么用](#chat)
- [上下文怎么选](#context)
- [思考开关](#thinking)
- [内置引擎](#engine)
- [配置存放与安全](#storage)
- [局限性](#limits)
- [常见问题](#faq)
- [兼容性](#compat)
- [反馈问题时请提供](#support)

<a id="what"></a>
## 它能解决什么

DSH 本身不带本地推理，只认「provider 适配器」。这个插件补上这一层：

- 一个 `.gguf` 文件 → 变成 DSH 模型选择器里的一个模型；
- 已有的 Ollama / LM Studio / vLLM / llama.cpp → 直接接进来；
- 不经过云端，请求只在本机回环地址上跑。

<a id="install"></a>
## 安装

| 项目 | 要求 |
| --- | --- |
| DSH | `>= 0.2.0 < 0.3.0` |
| Node.js | `>= 22.19` |
| 网络 | 仅在内置引擎下载时需要访问 GitHub |

从本地安装包：

```powershell
dsh plugin --profile web add "file:D:\path\to\dsh-local-model-0.4.1.tgz"
```

从 GitHub 安装：

```powershell
dsh plugin --profile web add github:<账号>/dsh-local-model
```

安装完成后：

1. **重启 DSH 一次**；
2. 刷新浏览器页面；
3. 打开 **设置 → 本地模型与 API**。

之后添加模型、修改配置都不需要再重启。

<a id="modes"></a>
## 四种接入方式

| 方式 | 适合谁 | 需要准备 | 上下文怎么生效 |
| --- | --- | --- | --- |
| **本地 GGUF** | 只有模型文件 | 主模型 `.gguf`（引擎可自动下载） | 作为 `--ctx-size` 启动引擎，同时告知 DSH |
| **已有 llama.cpp** | 已经跑着 `llama-server` | 服务地址（通常 `http://127.0.0.1:8080/v1`）、模型名 | 只影响 DSH 何时裁剪历史；真实容量由服务端启动参数决定 |
| **Ollama** | 用 Ollama 管理模型 | Ollama 在运行、模型已 pull | 通过原生接口 `options.num_ctx` 发送；改了会在下一次请求重新加载模型 |
| **模型 API** | 云端或兼容网关 | 根地址（含 `/v1`）、模型 ID、密钥 | 只影响 DSH 裁剪；不能超过服务商上限 |

> API 模式只实现 **OpenAI Chat Completions 兼容协议**。Anthropic Messages、Gemini、OpenAI Responses 需要额外的兼容网关。

### 示例：接入 Ollama

```yaml
- id: local-model
  config:
    models:
      - id: ollama-qwen
        backend: ollama
        baseURL: http://127.0.0.1:11434
        remoteModel: qwen3.5-9b-uncensored:latest
        contextWindow: 32768
        maxTokens: 2048
        think: false
```

设置页里更简单：接入方式选「Ollama」→ 填地址 → 点「读取服务端模型列表」→ 选模型 → 点「读取模型能力」自动填上下文。

### 示例：接入本地 GGUF

```yaml
- id: local-model
  config:
    models:
      - id: qwen-gguf
        backend: llamacpp
        file: C:\llama\models\qwen-uncensored-build\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
        ctxSize: 32768
        kvCacheType: q8_0
        gpuLayers: 999
        maxTokens: 2048
```

### 示例：接入云端 API

```yaml
- id: local-model
  config:
    models:
      - id: remote-api
        backend: openai
        baseURL: https://api.example.com/v1
        remoteModel: model-name
        apiKeyEnv: MODEL_API_KEY
        contextWindow: 32768
```

<a id="first-run"></a>
## 第一次使用（以本地 GGUF 为例）

1. 打开 **设置 → 本地模型与 API**。
2. 点 **添加模型**，接入方式选「本地 GGUF 文件」。
3. 点 **浏览本机 GGUF 文件**，进目录选主模型；也可以直接粘贴绝对路径。
4. 点 **读取模型信息**：插件读取 GGUF 头部，显示架构与模型声明的上下文，并填入建议值。
5. 按需调整上下文、KV 缓存精度、最大输出，点 **保存并生效**。
6. 回到对话，在模型选择器里选刚添加的模型。

第一次发消息会加载模型，通常几秒到几十秒；之后复用同一进程，空闲 10 分钟自动卸载。

<a id="settings"></a>
## 设置页逐项说明

### 推理引擎面板

页面顶部显示当前实际使用的 `llama-server`：

- **已就绪**：找到了可用引擎，旁边显示真实路径；
- **未安装**：可以点按钮下载（见 [内置引擎](#engine)）；
- 下载中显示进度条；失败会显示原因。

### 模型字段

| 字段 | 说明 |
| --- | --- |
| 模型 ID | 会话里提交的名字，插件内唯一，**保存后不能修改** |
| 显示名称 | 选择器里显示的名字 |
| 接入方式 | 本地 GGUF / 已有 llama.cpp / Ollama / 模型 API |
| GGUF 文件路径 | 主模型 `.gguf` 的绝对路径 |
| llama-server 可执行文件 | 可留空，插件会自动查找 |
| GPU 层数 | `999` = 尽量上 GPU；`0` = 纯 CPU |
| KV 缓存精度 | `f16` 精度最高；`q8_0` 省约一半显存；`q4_0` 最省 |
| Flash Attention | `auto` 交给引擎；选择 q8_0/q4_0 时自动开启 |
| 上下文长度 | 2K–256K 预设或自定义 |
| 最大输出 tokens | 单次回答上限，**思考内容也算在这里** |
| 单次请求超时 | 默认 300 秒 |
| 测试连接 | 不加载模型，只验证文件 / 服务 / 密钥是否可用 |

### Ollama 专属按钮

- **读取服务端模型列表**：从 `/api/tags` 拉取已安装模型；
- **读取模型能力**：从 `/api/show` 读取工具调用 / 思考 / 视觉能力、训练上下文、Modelfile 固定的 `num_ctx`，并给出建议上下文。

<a id="chat"></a>
## 在对话里怎么用

- 模型选择器里会出现一个分组（默认 **Local Models**），下面是添加的所有模型；
- 选中即用，和云端模型一样支持流式输出、工具调用、token 统计、中断；
- 工具调用是否触发，取决于模型自带的聊天模板（Qwen、Llama 3.1+、Mistral、Hermes 等通常支持）；
- 同一个模型可以被多个会话使用；不同模型各自一个进程，互不影响。

<a id="context"></a>
## 上下文怎么选

上下文是真实的显存 / 内存开销，不是越大越好：

| 上下文 | 适合 |
| --- | --- |
| 2K–8K | 小模型、短对话 |
| 16K–32K | 日常使用、带工具调用的 Agent |
| 64K 以上 | 需要大显存，8GB 卡基本放不下 |

**8GB 显存跑 9B Q4 的实测配方**：

| 设置 | 值 |
| --- | --- |
| 上下文 | `32768` |
| KV 缓存 | `q8_0` |
| Flash Attention | `auto`（自动 on） |
| 最大输出 | `2048` |

> 注意：DSH 的系统提示词、工具说明、AGENT.md 本身可能就有约 1.8 万 tokens。**16K 上下文连第一轮都放不下**，会直接报「上下文不足」。这不是模型或插件坏了，是提示词比上下文长。

<a id="thinking"></a>
## 思考开关

Qwen3.5 这类模型默认先输出思考，而思考**占用最大输出额度**：输出上限设成 48，回答会被思考吃光，界面上只剩一段思考。

- 不需要思考时，取消勾选「启用思考」；
- Ollama 会收到 `think: false`，托管 llama.cpp 会收到 `enable_thinking:false`；
- 云端 API 不受影响（不带这个字段）；
- 建议思考模型把最大输出设为 **2048 以上**。

<a id="engine"></a>
## 内置引擎（不用自己装 llama.cpp）

设置页点一下即可，插件会：

1. 查询 llama.cpp 官方最新 release；
2. 选择对应平台的预编译包（Windows/Linux/macOS，x64/arm64）；
3. 下载并用 GitHub 公布的 **SHA-256** 校验；
4. 解压到 `%USERPROFILE%\.dsh\local-model\engine\<tag>\`，写入清单，之后自动使用。

可选的包：

| 包 | 大小 | 说明 |
| --- | --- | --- |
| Vulkan（默认） | 约 35 MB | 自带运行库，NVIDIA / AMD / Intel 都能用 |
| CPU | 约 20 MB | 没有可用 GPU 时使用 |
| CUDA | 不内置 | 还需要几百 MB 的 CUDA 运行库，建议自己安装 llama.cpp |

引擎查找顺序（先找到先用）：

```
模型里填的 llamaServer
  → 全局 llamaServer
  → LLAMA_SERVER 环境变量
  → PATH
  → 常见解压目录（C:\llama、D:\llama、C:\llama.cpp、D:\llama.cpp、
     winget / scoop 链接、/usr/local/bin、/opt/homebrew/bin）
  → 插件下载的内置引擎
```

因此，已经装了 CUDA 版 llama.cpp 的机器不受影响，内置引擎只是兜底。面板上的「移除」可以随时删掉下载的引擎。

> 下载需要访问 GitHub。网络受限时，请手动安装 llama.cpp，解压到 `C:\llama` 这类目录即可被自动找到。

<a id="storage"></a>
## 配置存放与安全

- 模型列表存放在 `%USERPROFILE%\.dsh\local-model\<provider>.json`（未设置 `DSH_HOME` 时）；写入使用临时文件 + 原子替换，并用 `revision` 防止多页面并发覆盖；
- 密钥只保存在主机本地，读取设置不会回传；编辑时留空保留原值，点「清除保存的密钥」才删除；
- 设置接口要求 DSH 登录鉴权，并拒绝跨站请求；
- 引擎只监听 `127.0.0.1`，参数以数组传递，不经过 shell；
- 模型文件只读取头部（最多 32 MB）解析元数据，不复制、不上传；
- 删除该 JSON 并重启，可回到 YAML 里的初始配置。

<a id="limits"></a>
## 局限性

### 能力边界

- **只支持文本**：图片和附件会被替换成占位文字。模型即使声明了视觉能力，插件目前也不转发图片。
- **不下载模型**：GGUF 需要用户自己准备，插件只负责引擎。
- **一个插件实例一条 provider 路由**：所有模型在同一个分组下，不能拆成多个 provider。
- **不支持 embeddings / 重排**。
- **只支持 OpenAI 兼容协议**，不原生支持 Anthropic / Gemini / Responses。

### 性能边界

- **推理在进程外**：插件管理 `llama-server` 子进程，不是把推理搬进 Node。实测进程内方案只有 2–3 tokens/s，18K 提示词要几十分钟，因此没有采用。
- **首次请求慢**：需要加载模型（9B Q4 约 10–30 秒）。
- **上下文吃显存**：32K 以上要配合 q8_0 KV；64K 在 8GB 卡上基本不可行。
- **思考模型更费输出额度**：建议最大输出 ≥ 2048，或关闭思考。
- **多模型同时常驻会叠加内存**：可以把空闲卸载时间调短。

### 环境边界

- 需要 DSH Web 的设置页（依赖 `webServer` / `connection`）；纯 headless 部署只能用 YAML 配置，没有界面。
- 内置引擎下载依赖 GitHub 访问；网络受限时请手动安装 llama.cpp。
- CUDA 版引擎不内置；需要 CUDA 请自行安装 llama.cpp。
- 生成过程中保存设置会被拒绝（等当前回答结束再保存）。

### 交互边界

- 模型 ID 保存后不能修改，改名需要删除后重建。
- 上下文超限不会自动压缩重试，需要手动调大上下文或压缩会话。
- 视觉投影器（`mmproj-*.gguf`）不能当对话模型，插件会明确拒绝，请选择主模型。

<a id="faq"></a>
## 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 选择器里没有本地模型 | 没保存成功 / 没重启 | 设置页确认模型在列表里；首次安装后重启一次 |
| `LOCAL_RUNTIME_MISSING` | 找不到 llama-server | 点下载内置引擎，或填写 `llamaServer` 路径 |
| `LOCAL_RUNTIME_FAILED` | 显存不足 / 文件损坏 / 参数不支持 | 查看错误里的引擎输出；先试 `gpuLayers: 0`、降低上下文 |
| `CONTEXT_WINDOW_EXCEEDED` | 提示词超过上下文 | 调大上下文（32K）+ KV 选 q8_0；或压缩会话 |
| 选了 `mmproj-*.gguf` 被拒 | 那是视觉投影器 | 选择同目录下体积更大的主模型 |
| 工具调用不触发 | 模型没有 tools 模板 | 换支持工具调用的模型 |
| 保存报 409 | 正在生成 | 等回答结束再保存 |
| 下载引擎失败 | 网络 / GitHub 不可达 | 手动安装 llama.cpp，或配置代理后重试 |
| 回答只有思考没有正文 | 输出额度被思考吃光 | 最大输出调到 2048+，或关闭思考 |
| 首次对话卡很久 | 正在加载模型 | 正常；可把空闲卸载设为 0 常驻 |

<a id="compat"></a>
## 兼容性

| 项目 | 要求 |
| --- | --- |
| DSH | `>= 0.2.0 < 0.3.0` |
| Node.js | `>= 22.19` |
| 模型 | GGUF（llama.cpp / Ollama），或 OpenAI 兼容服务 |
| 平台 | Windows x64/arm64、Linux x64/arm64、macOS Intel / Apple Silicon |

开发、测试与发布说明见 [README.md](../README.md)，变更记录见 [CHANGELOG.md](../CHANGELOG.md)。

<a id="support"></a>
## 反馈问题时请提供

为了快速定位，请附上：

1. 操作系统与 DSH 版本（`dsh --version`）；
2. 接入方式（GGUF / 已有 llama.cpp / Ollama / API）；
3. 完整错误码与错误原文（界面上的红字）；
4. 模型文件路径或服务地址（**不要发 API Key**）；
5. 设置页「推理引擎」面板显示的路径；
6. 如果是上下文问题，附上上下文长度和最大输出设置。

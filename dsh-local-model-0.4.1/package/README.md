# dsh-local-model 0.4.1

在 **DSH 设置 → 本地模型与 API** 里选一个 GGUF 文件，或者接入 llama.cpp / Ollama / OpenAI 兼容 API，并设置这次要用的上下文长度。纯 JavaScript，没有前端构建步骤。

> 面向用户的完整说明（功能、四种接入、上下文、局限、FAQ）：[docs/USER-GUIDE.md](docs/USER-GUIDE.md)

当前版本针对真实的 Ollama 部署做过增强：读取模型自报的能力与上下文、按模型建议填上下文、一键关闭 thinking、保存前测试连接。

## 安装

要求 Node.js >=22.19、DSH >=0.2.0 <0.3.0。设置页依赖 DSH Web 的 ModuleLoader / `settings.section`，以及主机 `webServer` / `connection` 服务。

```powershell
dsh plugin --profile web add "C:\path\to\dsh-local-model-0.4.1.tgz"
```

也可以先 `npm install` 再 `dsh plugin --profile web add "link:C:\path\to\dsh-local-model"`。

安装后重启 DSH、刷新页面一次，进入 **设置 → 本地模型与 API**。之后保存模型不需要重启。

## 四种接入方式

| 方式 | 设置内容 | 上下文的作用 |
| --- | --- | --- |
| 本地 GGUF | 浏览选择 `.gguf`，可选 llama-server 路径 | 作为 `--ctx-size` 启动引擎，同时告知 DSH |
| 已有 llama.cpp | `/v1` 地址、服务端模型名 | DSH 历史预算；真实容量由服务端启动参数决定 |
| Ollama | 服务根地址、模型名（可从列表选） | 原生 `/api/chat` 发送 `options.num_ctx`，同时告知 DSH |
| 模型 API | OpenAI 兼容根地址、模型 ID、可选密钥 | DSH 历史预算；不能超过服务商硬上限 |

API 模式实现 **OpenAI Chat Completions 兼容协议**；Anthropic Messages、Gemini、OpenAI Responses 需要兼容网关。

## 快速开始：接 Ollama（当前部署）

1. 确认 Ollama 在跑：`ollama list` 能看到模型。
2. 打开 **设置 → 本地模型与 API**。列表为空时，页面会自动检测 `127.0.0.1:11434`，直接列出已安装模型；点一个就开始填表。
3. 点 **读取模型能力**：插件调用 Ollama `/api/show`，显示这个模型是否支持工具调用 / 思考 / 视觉，并把上下文填成建议值。
4. 按需调整 **上下文长度** 与 **最大输出 tokens**，点 **保存并生效**。
5. 回到对话，在模型选择器里选刚添加的模型。

本机实测（`qwen3.5-9b-uncensored:latest`）：

```
capabilities : completion, vision, tools, thinking
trained ctx  : 262144        Modelfile 固定 num_ctx（32k 变体）: 32768
建议上下文    : 32768
think:false  : 20.0 s，usage 17/41，正常回答
工具调用      : get_weather({"city":"杭州"})
```

### 关于思考（thinking）

qwen3.5 这类模型默认会先输出思考。思考内容**同样占用最大输出 tokens**：把输出上限设成 48，回答会被思考吃光，界面上只剩一段思考。插件为此提供了每模型的 **启用思考** 开关：

- 关闭后 Ollama 收到 `think: false`，响应更快、更省 token；
- 托管的 llama-server 收到 `chat_template_kwargs.enable_thinking: false`；
- 云端 API 不受影响（不带这个字段）。

建议思考模型把输出上限设为 **2048 以上**。

### 关于上下文长度

上下文不是越大越好，它是真实的 KV cache 内存 / 显存：

- 页面提供 2K / 4K / 8K / 16K / 32K / 64K / 128K / 256K 预设，也能手填。
- 超过 32K 会给出提示；Ollama 修改 `num_ctx` 后会在下一次请求重新加载模型。
- 对已有 llama.cpp 服务，这里只告诉 DSH 何时裁剪历史；真正的容量由服务端 `--ctx-size` 决定。

## 本地 GGUF（不需要 Ollama，也不需要自己装 llama.cpp）

这就是“直接用 GGUF”：插件读文件头、自己拉起引擎、把模型挂到 DSH，全程不经过 Ollama，也不转换权重。

1. 准备主模型 `.gguf`。
2. 选择“本地 GGUF 文件”，点 **浏览本机 GGUF 文件**，进入目录选择模型，或直接粘贴绝对路径。
3. 点 **读取模型信息**：读取 GGUF 头部，显示架构与声明上下文，按不超过 32K 的建议值填入。
4. 保存即可。进程按需启动、默认空闲十分钟卸载，只监听 `127.0.0.1`。

### 内置引擎（不用装 llama.cpp）

如果机器上没有任何 `llama-server`，设置页顶部的 **推理引擎** 面板会显示“未安装”，点一下就能装：

- 从 llama.cpp **官方 release** 下载对应平台的预编译包（默认 Vulkan，约 35 MB，自带运行库，NVIDIA/AMD/Intel 都能用）；
- 用 GitHub 公布的 **SHA-256** 校验下载内容；
- 解压到 `$DSH_HOME/local-model/engine/<tag>/`，写入清单，之后自动使用；
- 不需要管理员权限，不改 PATH，不动系统目录。

CPU 版约 20 MB，可在面板里选。CUDA 版没有内置：它还要额外几百 MB 的 CUDA 运行库，建议直接用你已有的 CUDA 安装（放在 `C:\llama` 这类目录即可被自动找到）。

引擎查找顺序（先找到先用）：模型级 `llamaServer` → 全局 `llamaServer` → `LLAMA_SERVER` 环境变量 → `PATH` → 常见解压目录（`C:\llama`、`D:\llama`、`C:\llama.cpp`、`D:\llama.cpp`、winget / scoop 链接、`/usr/local/bin`、`/opt/homebrew/bin`）→ **插件下载的内置引擎**。所以你已有的 CUDA 版永远优先，内置引擎只是兜底。

实测：Vulkan 包 35.3 MB，下载 + 校验 + 解压 6.8 秒；产物 `llama-server.exe --version` 返回 `0.7.0-dev (build 11873)`。

### 别选错文件：mmproj 不是模型

视觉模型的目录里通常有两个 `.gguf`：

```
Qwen3.5-9B-...-Q4_K_M.gguf          ← 主模型（5.6 GB），选这个
mmproj-Qwen3.5-9B-...-BF16.gguf     ← 视觉投影器（0.9 GB），不能单独对话
```

`mmproj` 只负责把图片变成向量，单独喂给 `llama-server` 会加载失败。插件现在会：

- 在浏览列表里把它标成“视觉投影（不能单独用）”；
- 选中后直接拒绝，并提示改用同目录的主模型；
- 对已保存的错误条目，请求时返回 `INVALID_REQUEST` 和同样的人话解释，而不是一段 llama.cpp 日志。

本机实测：`C:\llama\llama-server.exe` + `Qwen3.5-9B-Uncensored-...-Q4_K_M.gguf`，4K 上下文，10 秒就绪，正常回答；`mmproj-*.gguf` 被明确拒绝。

### 8GB 显存怎么跑 32K 上下文

DSH 的系统提示词、工具说明和 AGENT.md 加起来本身就有约 1.8 万 tokens，16K 上下文连第一轮都放不下（会看到 `CONTEXT_WINDOW_EXCEEDED`：`request (18648 tokens) exceeds the available context size (16384 tokens)`）。

9B Q4 权重约 5.6 GB，剩下的显存要留给 KV 缓存。推荐：

| 字段 | 值 |
| --- | --- |
| 上下文 | `32768` |
| KV 缓存精度 | `q8_0`（约省一半） |
| Flash Attention | `auto`（选了 q8_0 会自动加 `-fa on`） |

本机实测：32K + q8_0，19 秒就绪、正常回答。放不下就降到 16384，或 KV 选 `q4_0`。

对应的 YAML：

```yaml
- id: qwen-gguf
  backend: llamacpp
  file: C:\llama\models\qwen-uncensored-build\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
  ctxSize: 32768
  kvCacheType: q8_0
  gpuLayers: 999
  maxTokens: 2048
```

浏览的是 **DSH 主机磁盘**，只引用路径、不上传权重；远程访问时看到的是服务器文件。每目录最多显示 500 项。

## 保存、存储与安全

- 设置保存在 `$DSH_HOME/local-model/<provider>.json`（未设置 `DSH_HOME` 时用 `~/.dsh`），首次以 YAML 里的 `models` 为初始值，保存后 JSON 优先。
- 写入使用临时文件 + 原子替换；`revision` 防止多个页面互相覆盖。
- 生成期间保存会被拒绝，避免替换正在使用的引擎。
- API 读取不会回传密钥；编辑时留空保留原值，点“清除保存的密钥”才移除。密钥以明文存于主机本地 JSON，推荐用 `apiKeyEnv` 引用环境变量。
- 设置接口要求 DSH 身份验证，并拒绝跨站请求；`/browse` 只在已登录的页面内可用。
- 删除该 JSON 并重启，可以回到 YAML 初始配置。

## 高级配置（无界面 YAML）

```yaml
- id: local-model
  config:
    provider: local
    providerName: Local Models
    # stateFile: D:\dsh-data\web-local-models.json
    models:
      - id: local-gguf
        file: D:\models\model.gguf
        llamaServer: D:\llama\llama-server.exe
        ctxSize: 8192
        maxTokens: 2048
      - id: ollama-qwen
        backend: ollama
        baseURL: http://127.0.0.1:11434
        remoteModel: qwen3.5-9b-uncensored:latest
        contextWindow: 32768
        maxTokens: 2048
        think: false
      - id: remote-api
        backend: openai
        baseURL: https://api.example.com/v1
        remoteModel: model-name
        apiKeyEnv: MODEL_API_KEY
        contextWindow: 32768
```

Provider 字段：`llamaServer`、`basePort`（8177）、`startupTimeoutMs`（180000）、`idleTimeoutMs`（600000；0 常驻）、`defaultMaxTokens`（4096）、`requestTimeoutMs`（300000）、`autoContextCeiling`（8192）、`stateFile`。

模型字段：`id`、`name`、`backend`（`llamacpp` / `ollama` / `openai`）、`file`、`baseURL`、`apiKey` / `apiKeyEnv`、`remoteModel`、`ctxSize` / `contextWindow`、`maxTokens`、`requestTimeoutMs`、`think`。

GGUF 另有：`llamaServer`、`gpuLayers`（999；0 为 CPU）、`threads`、`batchSize`、`parallel`（1）、`jinja`（true）、`chatTemplate`、`extraArgs`。`ctxSize` 是每槽容量，插件按 `--ctx-size = ctxSize × parallel` 启动；`extraArgs` 不能覆盖插件管理的地址、端口、模型、上下文等参数。

`extraBody` 可传额外请求字段：Ollama 的采样参数放在 `extraBody.options`，`keep_alive` 放在顶层；核心字段优先。

## 开发与验证

```powershell
npm install
npm test                         # 102 个测试，含客户端交互、HTTP API、真实 Cordis / LlmRuntime
node scripts/verify-ollama.mjs   # 对真实 Ollama 跑文本 + 工具调用（没有 Ollama 时自动跳过）
npm run pack:check
```

自动化使用合成 GGUF 与模拟服务，不代替真实 GPU 推理；0.3.x 另在本机用 `qwen3.5-9b-uncensored:latest`（Ollama）和 `Qwen3.5-9B-Uncensored-Q4_K_M.gguf`（llama.cpp 直连）跑通了文本与工具调用（见上文实测）。

已知限制：仅文本输入，图片 / 附件按文字占位（该模型虽声明 vision，插件暂未转发图片）；一个插件只驱动一个 provider 路由；不支持 embeddings。

参考：[llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)、[Ollama Chat API](https://docs.ollama.com/api/chat)、[Ollama Show API](https://docs.ollama.com/api)。变更见 [CHANGELOG.md](CHANGELOG.md)。MIT License。




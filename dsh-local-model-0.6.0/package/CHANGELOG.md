# 0.6.0

修复「本地模型输出很慢」：插件以前默认下载 Vulkan 引擎，在 NVIDIA 卡上读提示词极慢；Vulkan 安装还常把核显排在第一位。

- 引擎默认跟随显卡：检测到 NVIDIA 时下载 **CUDA 包 + cudart 运行库**（约 560 MB），其他机器继续用 Vulkan；`engineVariant` 可强制 `auto|cuda|vulkan|cpu`。
- 实测（RTX 5060，9B Q4）：Vulkan 生成 25 tok/s、提示词 2.3 tok/s；CUDA 生成 54 tok/s、提示词 72 tok/s。首轮 DSH 长提示词差距最大。
- 启动前跑一次 `llama-server --list-devices`，自动用 `--device` 选独显，避免 9B 跑在 AMD 780M 这类核显上；模型可填 `device` 覆盖。
- 引擎面板显示检测到的显卡、推荐引擎、当前模型实际后端（CUDA / VULKAN / CPU）；已装 Vulkan 的机器可一键「换成 CUDA 引擎」，CPU 运行会在模型卡片标红。
- CUDA 下载支持断线重试（最多 3 次）；cudart 与主包一起解压，运行库自动放到 `llama-server` 同目录。
- 119 个测试通过。

# 0.5.0

本地 GGUF 支持视觉（看图）：

- 模型新增 `mmproj` 字段：设置页可浏览、读取并预览视觉投影器（架构、输入尺寸、投影类型、嵌入维度），保存后模型卡片带「视觉」标记。
- 托管引擎启动时传 `--mmproj`；模型声明 `image` 输入；请求时从 DSH 附件服务取出图片字节，转 `data:` URL 走 OpenAI 兼容 `image_url`，图片不出本机。
- 图片只在用户消息里转发；工具结果与未物化的图片保持占位文字。
- 「已有 llama.cpp」服务（`service: llama`）同样支持关闭思考，避免 Qwen3.5 的思考吃光输出。
- 实测：Qwen3.5-9B Q4 + BF16 投影器，360×180 图片正确读出 `HELLO 42`（91 prompt / 6 completion tokens）；112 个测试通过。

# 0.4.1

- 新增面向用户的完整说明书 [`docs/USER-GUIDE.md`](docs/USER-GUIDE.md)：功能、四种接入方式、上下文与思考、内置引擎、局限、FAQ、反馈清单；随 npm 包一起发布。
- README 中英文版增加用户指南入口。
- 无运行时变更。

# 0.4.0

GGUF 不再要求用户自己安装 llama.cpp：

- 设置页新增「推理引擎」面板：显示当前实际使用的 `llama-server`（显式路径 / PATH / 常见目录 / 内置），没有时可一键下载。
- 从 llama.cpp 官方 release 下载对应平台预编译包（默认 Vulkan 约 35 MB，自带运行库；CPU 约 20 MB），用 GitHub 公布的 SHA-256 校验，解压到 `$DSH_HOME/local-model/engine/<tag>/` 并写入清单，之后自动使用；可随时移除。
- 支持 zip（Windows）与 tar.gz（Linux/macOS），拒绝包含 `..` 的路径穿越条目，下载上限 800 MB。
- 引擎优先级排在所有用户安装之后：已有的 CUDA 版 `C:\llama` 仍然优先，内置引擎只作兜底。
- 实测：35.3 MB / 6.8 秒完成下载、校验与解压，产物 `llama-server.exe --version` 返回 `0.7.0-dev (build 11873)`；102 个测试通过。

# 0.3.2

让 8GB 显存真正跑得动长上下文：

- 新增两个 GGUF 字段与设置页下拉：`kvCacheType`（f16 / q8_0 / q4_0 → `-ctk/-ctv`）和 `flashAttention`（auto / on / off → `-fa`）。选 q8_0/q4_0 时自动开启 Flash Attention，显式 off 仍然优先。
- 上下文超限的错误改成可操作的中文提示：解析 llama.cpp 的 `request (N tokens) exceeds the available context size (M tokens)`，告诉用户本次请求多少、上限多少、建议调到多少，并提示 q8_0 + Flash Attention。
- 实测：`C:\llama` 的 9B Q4 + 32K 上下文 + q8_0 KV，19 秒就绪并正常回答；95 个测试通过。

# 0.3.1

直接 GGUF 的两个实际改进：

- `llama-server` 自动在常见解压目录里查找：`C:\llama`、`D:\llama`、`C:\llama.cpp`、`D:\llama.cpp`、winget 链接、scoop shims、`/usr/local/bin`、`/opt/homebrew/bin`。解压即用，不必改 PATH。
- 识别视觉投影器：`mmproj-*.gguf`（或 clip 架构且没有上下文长度）在浏览列表里标注“视觉投影（不能单独用）”，选中即拒绝并提示改用同目录主模型；已保存的错误条目请求时返回 `INVALID_REQUEST`，而不是 llama.cpp 的加载日志。
- 读取 GGUF 信息时返回找到的 `llama-server` 路径，设置页自动填入。
- 实测：`C:\llama\llama-server.exe` + `Qwen3.5-9B-Uncensored-...-Q4_K_M.gguf`，4K 上下文 10 秒就绪并正常回答；91 个测试通过。

# 0.3.0

针对真实 Ollama 部署（`qwen3.5-9b-uncensored:latest`）的增强：

- 设置页空列表时自动检测本机 Ollama（127.0.0.1:11434），直接列出已安装模型，一键开始填写。
- 新增「读取模型能力」：调用 Ollama `/api/show`，显示工具调用 / 思考 / 视觉能力、训练上下文、Modelfile 固定 `num_ctx`，并按建议值填好上下文（上限 32K）。
- 每模型新增「启用思考」开关：Ollama 发送 `think:false`；托管 llama-server 发送 `chat_template_kwargs.enable_thinking:false`；云端 API 不带该字段。默认输出上限从 1024 调到 2048，并提示思考会占用输出额度。
- 新增「测试连接」：GGUF 校验文件与 llama-server；Ollama 校验 `/api/tags` 与模型是否存在；API 校验 `/models`。
- GGUF / Ollama 读取后按建议值填上下文（不超过 32K），超过 32K 给出内存 / 显存提示；上下文预设扩展到 2K–256K。
- 客户端设置区注册改用 `ctx.effect`，插件卸载时正确收回。
- 实测：文本请求 20.0 s、usage 17/41；工具调用返回 `get_weather({"city":"杭州"})`；86 个自动化测试通过。

# 0.2.0

- 新增 DSH 设置 → 本地模型与 API，支持添加、编辑、删除模型，保存即时生效。
- GGUF 主机文件浏览、元数据校验，无需上传权重。
- 提供本地 GGUF、已有 llama.cpp、原生 Ollama、OpenAI 兼容 API 四种接入选项。
- Ollama 原生流支持 thinking、工具调用、用量统计，通过 options.num_ctx 设置实际上下文。
- 每模型可选预设 / 自定义上下文、输出上限、超时、模型名和认证方式。
- 模型配置原子持久化；读取时密钥脱敏；支持保留和显式清除；revision 防止并发覆盖。
- 修复多模型冷启动端口竞争；进程退出前保持端口预留。
- 修复模板文件参数，使用 --chat-template-file。
- 修复模型级 llamaServer 缺少 schema 声明。
- 明确 --parallel，按每槽上下文分配总 --ctx-size。
- 拒绝 extraArgs 覆盖插件管理的关键参数。
- 增加请求超时并区分 TIMEOUT / ABORTED，限制过大流事件。
- 修复工具调用后到 ID 导致流内身份不一致的问题。

验证使用模拟服务及真实 Cordis / LlmRuntime 集成；未实际加载 GPU 模型或在安装中的桌面页面验收。

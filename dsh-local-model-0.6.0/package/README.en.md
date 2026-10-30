# dsh-local-model 0.6.0

Configure GGUF files, existing llama.cpp / Ollama servers and OpenAI-compatible APIs in **DSH Settings → 本地模型与 API**, and pick the context length each one runs with. The page is Chinese; the code and this note are English.

> End-user guide (features, four modes, context, limits, FAQ): [docs/USER-GUIDE.md](docs/USER-GUIDE.md)

Requires Node >=22.19, DSH >=0.2.0 <0.3.0, and DSH's web ModuleLoader / `settings.section` plus authenticated `webServer` / `connection` services. GGUF additionally needs a separately installed `llama-server` and weights.

Install: `dsh plugin --profile web add "C:\path\dsh-local-model-0.6.0.tgz"`. Restart DSH and refresh the client once; later model saves apply immediately.

## What 0.3.x adds for Ollama

- **Auto-detect**: an empty list probes `127.0.0.1:11434` and offers the installed models as one-click additions.
- **Read model facts**: `/api/show` reports capabilities (tools / thinking / vision), the trained context, a Modelfile-pinned `num_ctx`, and a suggested context capped at 32K.
- **Thinking switch per model**: `think: false` for Ollama, `chat_template_kwargs.enable_thinking: false` for a managed llama-server, nothing for cloud APIs. Thinking consumes the output budget, so the default cap is now 2048.
- **Test connection** before saving: GGUF file + `llama-server` presence, Ollama `/api/tags`, or the API's `/models`.

Verified against `qwen3.5-9b-uncensored:latest` on this machine: capabilities `completion, vision, tools, thinking`; trained context 262144; a `think:false` answer in 20 s (usage 17/41); a real `get_weather({"city":"杭州"})` tool call.

## Modes and context

Managed GGUF sets per-slot context with explicit parallelism (`--ctx-size = ctxSize × parallel`). Native Ollama sends `options.num_ctx` on `/api/chat`; changing it reloads the model on the next request. Existing llama.cpp/API context settings only guide DSH history management — they cannot raise server capacity or model limits.

Presets run 2K…256K with a warning above 32K; anything above that is real KV-cache memory.

## Storage and safety

Settings persist atomically in `$DSH_HOME/local-model/<provider>.json` (default `~/.dsh`); YAML seeds the first run, saved models then take priority. API keys are redacted on read, preserved when the field is left untouched, and cleared explicitly. Settings writes require DSH authentication and reject cross-site requests; saves are refused while a generation is active. Delete the JSON and restart to return to YAML defaults.

## Development

```powershell
npm install
npm test                         # 119 tests, including client interaction, HTTP API, real Cordis / LlmRuntime
node scripts/verify-ollama.mjs   # live text + tool call against a real Ollama; skips when unavailable
npm run pack:check
```

Tests use synthetic GGUF and mock inference services; real GPU inference is verified separately (see above). Text-only input: images and attachments become placeholders.

See [README.md](README.md) for the full configuration reference and [CHANGELOG.md](CHANGELOG.md) for changes. MIT License.





/**
 * Minimal GGUF header reader.
 *
 * A GGUF file states the facts this plugin needs — architecture, trained
 * context length, layer count — in its metadata block, before any tensor data.
 * Reading them means a model can be configured with nothing but a file path:
 * the context window DSH advertises then matches what the model was actually
 * trained for instead of a guess.
 *
 * Only the header is read, and only a bounded prefix of the file, so pointing
 * at a 40 GB model costs a few megabytes at most. Unknown metadata is skipped.
 * A file that ends mid-header (or a prefix that stops before the keys we want)
 * yields whatever was decoded plus `complete: false`; callers fall back to a
 * default context size rather than failing the model.
 *
 * @module dsh-local-model/gguf
 */

import { open } from 'node:fs/promises'

/** `GGUF` read as a little-endian uint32. */
const GGUF_MAGIC = 0x46554747

/** Metadata is always near the start; 32 MiB covers large tokenizer tables. */
const DEFAULT_MAX_BYTES = 32 * 1024 * 1024

/** GGUF metadata value type tags. */
const TYPE = {
  UINT8: 0,
  INT8: 1,
  UINT16: 2,
  INT16: 3,
  UINT32: 4,
  INT32: 5,
  FLOAT32: 6,
  BOOL: 7,
  STRING: 8,
  ARRAY: 9,
  UINT64: 10,
  INT64: 11,
  FLOAT64: 12,
}

/** Fixed byte width of each scalar type, when it has one. */
const SCALAR_WIDTH = {
  [TYPE.UINT8]: 1,
  [TYPE.INT8]: 1,
  [TYPE.UINT16]: 2,
  [TYPE.INT16]: 2,
  [TYPE.UINT32]: 4,
  [TYPE.INT32]: 4,
  [TYPE.FLOAT32]: 4,
  [TYPE.BOOL]: 1,
  [TYPE.UINT64]: 8,
  [TYPE.INT64]: 8,
  [TYPE.FLOAT64]: 8,
}

/** Thrown internally when the bounded read ends inside the metadata block. */
class Truncated extends Error {}

/**
 * Explain why a `.gguf` file cannot be a chat model on its own, if it cannot.
 *
 * A vision build ships two files: the language model and an `mmproj-*.gguf`
 * projector that only turns pixels into embeddings. The projector is a normal
 * GGUF with a `clip` architecture and no context length, so a file picker that
 * lists every `.gguf` invites exactly this mistake — and `llama-server` answers
 * it with an opaque load failure.
 * @param {string} file - the configured path.
 * @param {{ architecture?: string, contextLength?: number }} [metadata] - its GGUF header.
 * @returns {string | undefined} an actionable warning, or undefined for a real model.
 */
export function mmprojWarning(file, metadata) {
  const base = String(file ?? '').split(/[\\/]/).pop() ?? ''
  if (/^mmproj[-_.]/i.test(base)) {
    return '这是视觉投影器（mmproj）文件，不能单独作为对话模型；请选择同目录下的主模型 .gguf（通常文件名不带 mmproj，体积也大得多）'
  }
  if (metadata?.architecture === 'clip' && metadata?.contextLength === undefined) {
    return '这个 GGUF 是视觉投影器（clip 架构、没有上下文长度），不能单独作为对话模型；请选择主模型文件'
  }
  return undefined
}

/**
 * Read one GGUF file's metadata.
 * @param {string} file - path to the `.gguf` file.
 * @param {{ maxBytes?: number }} [options] - read bound.
 * @returns {Promise<ReturnType<typeof parseGgufHeader>>} decoded metadata; never throws for a truncated prefix.
 * @throws {Error} when the file cannot be read or is not GGUF at all.
 */
export async function readGgufMetadata(file, options = {}) {
  const handle = await open(file, 'r')
  try {
    const stat = await handle.stat()
    if (!stat.isFile()) throw new Error(`${file} is not a regular file`)
    if (stat.size < 24) throw new Error(`${file} is too small to be a GGUF file`)
    const length = Math.min(stat.size, options.maxBytes ?? DEFAULT_MAX_BYTES)
    const buffer = Buffer.allocUnsafe(length)
    const { bytesRead } = await handle.read(buffer, 0, length, 0)
    return parseGgufHeader(buffer.subarray(0, bytesRead))
  } finally {
    await handle.close()
  }
}

/**
 * Decode a GGUF header from an in-memory prefix.
 * @param {Buffer} buffer - the file prefix, starting at byte 0.
 * @returns {{ version: number, tensorCount: number, kvCount: number, architecture?: string,
 *   name?: string, contextLength?: number, embeddingLength?: number, blockCount?: number,
 *   complete: boolean }} decoded metadata.
 * @throws {Error} when the magic or version is not GGUF.
 */
export function parseGgufHeader(buffer) {
  const reader = new Reader(buffer)
  const magic = reader.uint32()
  if (magic !== GGUF_MAGIC) throw new Error('not a GGUF file (bad magic)')
  const version = reader.uint32()
  if (version < 2 || version > 3) throw new Error(`unsupported GGUF version ${version}`)
  const tensorCount = reader.count()
  const kvCount = reader.count()

  /** @type {Map<string, string | number | boolean | bigint>} */
  const wanted = new Map()
  let complete = true
  try {
    for (let index = 0; index < kvCount; index += 1) {
      const key = reader.string()
      const type = reader.uint32()
      if (isWantedKey(key)) {
        const value = reader.value(type, key)
        if (value !== SKIPPED) wanted.set(key, value)
      } else {
        reader.skip(type)
      }
    }
  } catch (error) {
    if (!(error instanceof Truncated)) throw error
    complete = false
  }

  const architecture = asString(wanted.get('general.architecture'))
  const name = asString(wanted.get('general.name'))
  const contextLength = pickLength(wanted, architecture, 'context_length')
  const embeddingLength = pickLength(wanted, architecture, 'embedding_length')
  const blockCount = pickLength(wanted, architecture, 'block_count')

  return {
    version,
    tensorCount,
    kvCount,
    ...(architecture === undefined ? {} : { architecture }),
    ...(name === undefined ? {} : { name }),
    ...(contextLength === undefined ? {} : { contextLength }),
    ...(embeddingLength === undefined ? {} : { embeddingLength }),
    ...(blockCount === undefined ? {} : { blockCount }),
    complete,
  }
}

/** Sentinel for a wanted key whose value type we cannot use. */
const SKIPPED = Symbol('gguf.skipped')

/**
 * Whether one metadata key describes something this plugin uses.
 * @param {string} key - the metadata key.
 * @returns {boolean} true for architecture, display name, or a length fact.
 */
function isWantedKey(key) {
  return key === 'general.architecture'
    || key === 'general.name'
    || key.endsWith('.context_length')
    || key.endsWith('.embedding_length')
    || key.endsWith('.block_count')
}

/**
 * Read the length fact for one architecture, falling back to the first match.
 * @param {Map<string, string | number | boolean | bigint>} values - decoded values.
 * @param {string | undefined} architecture - the declared architecture.
 * @param {string} suffix - `context_length`, `embedding_length`, or `block_count`.
 * @returns {number | undefined} the positive integer value, when present.
 */
function pickLength(values, architecture, suffix) {
  const exact = architecture === undefined ? undefined : values.get(`${architecture}.${suffix}`)
  if (typeof exact === 'number' && exact > 0) return exact
  for (const [key, value] of values) {
    if (!key.endsWith(`.${suffix}`)) continue
    if (typeof value === 'number' && value > 0) return value
  }
  return undefined
}

/**
 * Narrow a decoded scalar to a string.
 * @param {unknown} value - decoded value.
 * @returns {string | undefined} the string, when it is one.
 */
function asString(value) {
  return typeof value === 'string' && value !== '' ? value : undefined
}

/** Bounds-checked little-endian cursor over a header prefix. */
class Reader {
  /** @param {Buffer} buffer - the header prefix. */
  constructor(buffer) {
    this.buffer = buffer
    this.offset = 0
  }

  /**
   * Require `width` more bytes.
   * @param {number} width - byte count about to be read.
   * @returns {number} the offset the read starts at.
   */
  need(width) {
    if (this.offset + width > this.buffer.length) throw new Truncated()
    const at = this.offset
    this.offset += width
    return at
  }

  /** @returns {number} the next uint32. */
  uint32() {
    return this.buffer.readUInt32LE(this.need(4))
  }

  /**
   * Read a bounded count field.
   * @returns {number} the count.
   */
  count() {
    const at = this.need(8)
    const value = this.buffer.readBigUInt64LE(at)
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('GGUF count out of range')
    return Number(value)
  }

  /** @returns {string} the next length-prefixed UTF-8 string. */
  string() {
    const at = this.need(8)
    const length = this.buffer.readBigUInt64LE(at)
    if (length > BigInt(this.buffer.length)) throw new Truncated()
    const bytes = this.need(Number(length))
    return this.buffer.toString('utf8', bytes, bytes + Number(length))
  }

  /**
   * Skip one value of any type.
   * @param {number} type - the value type tag.
   */
  skip(type) {
    if (type === TYPE.STRING) {
      this.string()
      return
    }
    if (type === TYPE.ARRAY) {
      const element = this.uint32()
      const count = this.count()
      if (element === TYPE.STRING) {
        for (let index = 0; index < count; index += 1) this.string()
        return
      }
      const width = SCALAR_WIDTH[element]
      if (width === undefined) throw new Error(`GGUF array element type ${element} is not supported`)
      this.need(width * count)
      return
    }
    const width = SCALAR_WIDTH[type]
    if (width === undefined) throw new Error(`GGUF value type ${type} is not supported`)
    this.need(width)
  }

  /**
   * Decode one value of a known type.
   * @param {number} type - the value type tag.
   * @param {string} key - metadata key (diagnostics only).
   * @returns {string | number | boolean | typeof SKIPPED} the decoded scalar.
   */
  value(type, key) {
    switch (type) {
      case TYPE.STRING:
        return this.string()
      case TYPE.BOOL: {
        const at = this.need(1)
        return this.buffer.readUInt8(at) !== 0
      }
      case TYPE.UINT8:
        return this.buffer.readUInt8(this.need(1))
      case TYPE.INT8:
        return this.buffer.readInt8(this.need(1))
      case TYPE.UINT16:
        return this.buffer.readUInt16LE(this.need(2))
      case TYPE.INT16:
        return this.buffer.readInt16LE(this.need(2))
      case TYPE.UINT32:
        return this.buffer.readUInt32LE(this.need(4))
      case TYPE.INT32:
        return this.buffer.readInt32LE(this.need(4))
      case TYPE.FLOAT32:
        return this.buffer.readFloatLE(this.need(4))
      case TYPE.FLOAT64:
        return this.buffer.readDoubleLE(this.need(8))
      case TYPE.UINT64: {
        const raw = this.buffer.readBigUInt64LE(this.need(8))
        return raw <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(raw) : SKIPPED
      }
      case TYPE.INT64: {
        const raw = this.buffer.readBigInt64LE(this.need(8))
        return raw >= BigInt(Number.MIN_SAFE_INTEGER) && raw <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(raw) : SKIPPED
      }
      default: {
        // Unknown types cannot be skipped safely; report and let the caller
        // keep what it already decoded.
        throw new Truncated(`unsupported GGUF metadata type ${type} for ${key}`)
      }
    }
  }
}

// HTTP API：导入语料、训练、查轮次、编码、解码、轨迹、版本对比、导出模型。
import * as http from 'node:http';
import * as net from 'node:net';
import { BpeError, Encoder, train, WordEntry } from './bpe';
import { Store } from './db';

interface Ctx {
  store: Store;
}

function send(res: http.ServerResponse, status: number, body: unknown): void {
  const json = JSON.stringify(body);
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(json);
}

function sendError(res: http.ServerResponse, err: unknown): void {
  if (err instanceof BpeError) {
    const status =
      err.code === 'MODEL_NOT_FOUND' || err.code === 'ROUND_NOT_FOUND' ? 404
      : err.code === 'NOT_TRAINED' ? 409
      : err.code === 'LIMIT_EXCEEDED' ? 413
      : 400;
    send(res, status, { error: { code: err.code, message: err.message } });
  } else {
    send(res, 500, { error: { code: 'INTERNAL', message: String(err) } });
  }
}

function readBody(req: http.IncomingMessage): Promise<any> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (c: Buffer) => {
      size += c.length;
      if (size > 16 * 1024 * 1024) reject(new BpeError('LIMIT_EXCEEDED', '请求体过大'));
      chunks.push(c);
    });
    req.on('end', () => {
      if (chunks.length === 0) return resolve({});
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')));
      } catch {
        reject(new BpeError('INVALID_PARAM', '请求体不是合法 JSON'));
      }
    });
    req.on('error', reject);
  });
}

function parseWords(body: any): WordEntry[] {
  // 支持 {words:[{word,weight}]} 或 {text:"word weight\\n..."}
  if (Array.isArray(body?.words)) {
    return body.words.map((w: any) => {
      if (typeof w !== 'object' || w === null) throw new BpeError('INVALID_PARAM', 'words 元素格式错误');
      return { word: w.word, weight: w.weight };
    });
  }
  if (typeof body?.text === 'string') {
    const out: WordEntry[] = [];
    for (const line of body.text.split(/\r?\n/)) {
      const t = line.trim();
      if (!t) continue;
      const m = t.match(/^(\S+)\s+(\d+)$/);
      if (!m) throw new BpeError('INVALID_PARAM', `语料行格式错误（应为 "单词 正整数词频"）: ${t}`);
      out.push({ word: m[1], weight: Number(m[2]) });
    }
    return out;
  }
  throw new BpeError('INVALID_PARAM', '需要 words 数组或 text 文本语料');
}

function requireWordsList(body: any): string[] {
  if (!Array.isArray(body?.words) || body.words.length === 0 || body.words.some((w: any) => typeof w !== 'string')) {
    throw new BpeError('INVALID_PARAM', '需要非空字符串数组 words');
  }
  return body.words;
}

async function route(ctx: Ctx, req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://localhost');
  const parts = url.pathname.split('/').filter(Boolean);
  const method = req.method ?? 'GET';
  const { store } = ctx;

  if (method === 'GET' && url.pathname === '/health') {
    return send(res, 200, { ok: true });
  }
  if (method === 'GET' && url.pathname === '/models') {
    return send(res, 200, { models: store.listModels() });
  }
  if (method === 'POST' && url.pathname === '/corpus/import') {
    const body = await readBody(req);
    if (typeof body?.name !== 'string' || !body.name) {
      throw new BpeError('INVALID_PARAM', '需要模型名 name');
    }
    const words = parseWords(body);
    const model = store.importCorpus(body.name, words);
    return send(res, 201, { model });
  }
  if (parts[0] === 'models' && parts.length >= 2) {
    const id = Number(parts[1]);
    if (!Number.isInteger(id)) throw new BpeError('INVALID_PARAM', '模型 ID 必须为整数');

    if (method === 'GET' && parts.length === 2) {
      return send(res, 200, { model: store.getModel(id) });
    }
    if (method === 'POST' && parts[2] === 'train' && parts.length === 3) {
      const body = await readBody(req);
      const maxMerges = body.maxMerges ?? 100;
      const minCount = body.minCount ?? 1;
      const corpus = store.getCorpus(id);
      if (corpus.length === 0) throw new BpeError('INVALID_PARAM', '语料为空');
      let result;
      try {
        result = train(corpus, maxMerges, minCount);
        store.saveTrainedModel(id, result, maxMerges, minCount);
      } catch (err) {
        store.markFailed(id);
        throw err;
      }
      return send(res, 200, {
        model: store.getModel(id),
        stopReason: result.stopReason,
        rounds: result.merges.length,
        vocabSize: result.symbols.length,
      });
    }
    if (method === 'GET' && parts[2] === 'rounds' && parts.length === 3) {
      const page = Math.max(1, Number(url.searchParams.get('page') ?? 1));
      const pageSize = Math.min(200, Math.max(1, Number(url.searchParams.get('pageSize') ?? 20)));
      return send(res, 200, store.listRounds(id, page, pageSize));
    }
    if (method === 'GET' && parts[2] === 'rounds' && parts.length === 4) {
      return send(res, 200, { round: store.getRound(id, Number(parts[3])) });
    }
    if (method === 'POST' && parts[2] === 'encode' && parts.length === 3) {
      const body = await readBody(req);
      const words = requireWordsList(body);
      const enc = new Encoder(store.loadModelData(id));
      return send(res, 200, { encodings: words.map((w) => ({ word: w, ids: enc.encode(w) })) });
    }
    if (method === 'POST' && parts[2] === 'decode' && parts.length === 3) {
      const body = await readBody(req);
      if (!Array.isArray(body?.ids)) throw new BpeError('INVALID_PARAM', '需要 ids（数组的数组）');
      const enc = new Encoder(store.loadModelData(id));
      const decoded = body.ids.map((row: any) => {
        if (!Array.isArray(row) || row.some((x: any) => !Number.isInteger(x))) {
          throw new BpeError('INVALID_PARAM', 'ids 必须为整数数组的数组');
        }
        return enc.decode(row);
      });
      return send(res, 200, { words: decoded });
    }
    if (method === 'GET' && parts[2] === 'trace' && parts.length === 3) {
      const word = url.searchParams.get('word');
      if (!word) throw new BpeError('INVALID_PARAM', '需要查询参数 word');
      const enc = new Encoder(store.loadModelData(id));
      return send(res, 200, { word, steps: enc.trace(word) });
    }
    if (method === 'GET' && parts[2] === 'export' && parts.length === 3) {
      return send(res, 200, store.exportModel(id));
    }
  }
  if (method === 'POST' && url.pathname === '/compare') {
    const body = await readBody(req);
    const a = Number(body?.a);
    const b = Number(body?.b);
    if (!Number.isInteger(a) || !Number.isInteger(b)) {
      throw new BpeError('INVALID_PARAM', '需要整数模型 ID: a 与 b');
    }
    const words = requireWordsList(body);
    const encA = new Encoder(store.loadModelData(a));
    const encB = new Encoder(store.loadModelData(b));
    return send(res, 200, {
      comparisons: words.map((w) => ({
        word: w,
        a: { model: a, ids: encA.encode(w) },
        b: { model: b, ids: encB.encode(w) },
      })),
    });
  }
  send(res, 404, { error: { code: 'NOT_FOUND', message: `${method} ${url.pathname} 不存在` } });
}

export function findFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const port = (srv.address() as net.AddressInfo).port;
      srv.close(() => resolve(port));
    });
  });
}

export async function startServer(store: Store, port?: number): Promise<{ server: http.Server; port: number }> {
  const p = port ?? (await findFreePort());
  const server = http.createServer((req, res) => {
    route({ store }, req, res).catch((err) => sendError(res, err));
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(p, '127.0.0.1', () => resolve());
  });
  return { server, port: p };
}

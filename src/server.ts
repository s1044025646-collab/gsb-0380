import * as http from "node:http";
import { AddressInfo } from "node:net";
import { BpeError, decodeIds, encodeWord, traceWord, LIMITS } from "./core";
import { Store, defaultDbPath } from "./store";

function json(res: http.ServerResponse, status: number, body: unknown) {
  const s = JSON.stringify(body);
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(s);
}

function readBody(req: http.IncomingMessage): Promise<any> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (c: Buffer) => {
      size += c.length;
      if (size > 10 * 1024 * 1024) { reject(new BpeError("LIMIT_EXCEEDED", "请求体超过 10MB")); req.destroy(); return; }
      chunks.push(c);
    });
    req.on("end", () => {
      if (chunks.length === 0) return resolve({});
      try { resolve(JSON.parse(Buffer.concat(chunks).toString("utf8"))); }
      catch { reject(new BpeError("INVALID_INPUT", "请求体不是合法 JSON")); }
    });
    req.on("error", reject);
  });
}

function parseWordsField(body: any): string[] {
  const words = body?.words;
  if (!Array.isArray(words) || words.length === 0 || words.some((w) => typeof w !== "string" || w.length === 0))
    throw new BpeError("INVALID_INPUT", "words 必须是非空字符串数组");
  if (words.length > LIMITS.MAX_WORDS) throw new BpeError("LIMIT_EXCEEDED", "词数超过上限");
  return words;
}

function parseIdsField(body: any): number[][] {
  const ids = body?.ids;
  if (!Array.isArray(ids) || ids.length === 0 || ids.some((a) => !Array.isArray(a) || a.some((n) => !Number.isInteger(n))))
    throw new BpeError("INVALID_INPUT", "ids 必须是整数数组的数组（外层数组保留词边界）");
  return ids;
}

export function createServer(store: Store): http.Server {
  return http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? "/", "http://localhost");
      const p = url.pathname;
      const body = req.method === "POST" ? await readBody(req) : {};
      const vMatch = p.match(/^\/versions\/(\d+)(\/(rounds|encode|decode|trace|export|corpus))?(\/(\d+))?$/);

      if (p === "/health") return json(res, 200, { ok: true });
      if (p === "/versions" && req.method === "GET") return json(res, 200, { versions: store.listVersions() });
      if (p === "/train" && req.method === "POST") {
        const maxMerges = body.maxMerges ?? 100;
        const minCount = body.minCount ?? 1;
        const { version, result } = store.trainNewVersion(body.words, { maxMerges, minCount });
        return json(res, 200, { version, stopReason: result.stopReason, vocabSize: result.vocabSize, rounds: result.rounds.length, totalWeight: result.totalWeight });
      }
      if (p === "/compare" && req.method === "POST") {
        const v1 = Number(body.v1), v2 = Number(body.v2);
        if (!Number.isInteger(v1) || !Number.isInteger(v2)) throw new BpeError("INVALID_INPUT", "v1/v2 必须是整数版本号");
        const m1 = store.getModel(v1), m2 = store.getModel(v2);
        const words = parseWordsField(body);
        return json(res, 200, { rows: words.map((w) => ({ word: w, v1: encodeWord(w, m1), v2: encodeWord(w, m2) })) });
      }
      if (vMatch) {
        const version = Number(vMatch[1]);
        const sub = vMatch[3];
        if (!sub && req.method === "GET") return json(res, 200, store.getVersion(version));
        if (sub === "corpus" && req.method === "GET") return json(res, 200, { version, corpus: store.getCorpus(version) });
        if (sub === "export" && req.method === "GET") return json(res, 200, store.exportModel(version));
        if (sub === "rounds" && vMatch[5] && req.method === "GET") return json(res, 200, store.getRound(version, Number(vMatch[5])));
        if (sub === "rounds" && req.method === "GET") {
          const page = Math.max(1, Number(url.searchParams.get("page") ?? 1));
          const pageSize = Math.min(200, Math.max(1, Number(url.searchParams.get("pageSize") ?? 20)));
          return json(res, 200, { version, page, pageSize, ...store.getRounds(version, page, pageSize) });
        }
        if (sub === "encode" && req.method === "POST") {
          const model = store.getModel(version);
          const words = parseWordsField(body);
          return json(res, 200, { version, tokens: words.map((w) => encodeWord(w, model)) });
        }
        if (sub === "decode" && req.method === "POST") {
          const model = store.getModel(version);
          const ids = parseIdsField(body);
          return json(res, 200, { version, words: ids.map((a) => decodeIds(a, model)) });
        }
        if (sub === "trace" && req.method === "GET") {
          const word = url.searchParams.get("word");
          if (!word) throw new BpeError("INVALID_INPUT", "缺少 word 参数");
          return json(res, 200, { version, word, ...traceWord(word, store.getModel(version)) });
        }
      }
      throw new BpeError("NOT_FOUND", `${req.method} ${p} 不存在`);
    } catch (e: any) {
      if (e instanceof BpeError) return json(res, e.code === "NOT_FOUND" ? 404 : e.code === "VERSION_NOT_FOUND" || e.code === "ROUND_NOT_FOUND" ? 404 : 400, { error: { code: e.code, message: e.message } });
      return json(res, 500, { error: { code: "INTERNAL", message: String(e?.message ?? e) } });
    }
  });
}

if (require.main === module) {
  const portArg = process.argv.find((a) => a.startsWith("--port="));
  const port = portArg ? Number(portArg.split("=")[1]) : Number(process.env.PORT ?? 0); // 默认选空闲端口
  const store = new Store(defaultDbPath());
  const server = createServer(store);
  server.listen(port, () => {
    const addr = server.address() as AddressInfo;
    console.log(`BPE API 已启动: http://127.0.0.1:${addr.port} (数据目录: ./data)`);
  });
}

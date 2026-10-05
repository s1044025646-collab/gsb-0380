import { decodeIds, encodeWord, symbolTexts, traceWord } from "./core";
import { Store, defaultDbPath } from "./store";

function parseWordsArg(s: string): { word: string; freq: number }[] {
  // 格式: "low:5,lower:2" 或纯词（频数 1）
  return s.split(",").map((part) => {
    const i = part.lastIndexOf(":");
    if (i > 0 && /^\d+$/.test(part.slice(i + 1))) return { word: part.slice(0, i), freq: Number(part.slice(i + 1)) };
    return { word: part, freq: 1 };
  });
}

function usage() {
  console.log(`用法: node dist/src/cli.js <命令> [参数]
  train --words "low:5,lower:2" [--max-merges 100] [--min-count 1]
  versions
  rounds --version 1 [--page 1] [--page-size 20]
  round --version 1 --round 3
  encode --version 1 --words "low,lower"
  decode --version 1 --ids "1,2;3,4"     (分号分隔词，保留词边界)
  trace --version 1 --word lower
  compare --v1 1 --v2 2 --words "low,lower"
  export --version 1 [--out model.json]
  corpus --version 1
  serve [--port 0]`);
}

const args = process.argv.slice(2);
const cmd = args[0];
function opt(name: string, def?: string): string | undefined {
  const a = args.find((x) => x.startsWith(`--${name}=`));
  return a ? a.split("=").slice(1).join("=") : def;
}

const store = new Store(defaultDbPath());
try {
  switch (cmd) {
    case "train": {
      const words = parseWordsArg(opt("words") ?? (() => { throw new Error("缺少 --words"); })());
      const { version, result } = store.trainNewVersion(words, {
        maxMerges: Number(opt("max-merges", "100")),
        minCount: Number(opt("min-count", "1")),
      });
      console.log(JSON.stringify({ version, stopReason: result.stopReason, vocabSize: result.vocabSize, rounds: result.rounds.length }, null, 2));
      break;
    }
    case "versions": console.log(JSON.stringify(store.listVersions(), null, 2)); break;
    case "rounds": {
      const v = Number(opt("version"));
      console.log(JSON.stringify(store.getRounds(v, Number(opt("page", "1")), Number(opt("page-size", "20"))), null, 2));
      break;
    }
    case "round": console.log(JSON.stringify(store.getRound(Number(opt("version")), Number(opt("round"))), null, 2)); break;
    case "encode": {
      const model = store.getModel(Number(opt("version")));
      const words = (opt("words") ?? "").split(",");
      console.log(JSON.stringify(words.map((w) => ({ word: w, ids: encodeWord(w, model) })), null, 2));
      break;
    }
    case "decode": {
      const model = store.getModel(Number(opt("version")));
      const groups = (opt("ids") ?? "").split(";").filter(Boolean).map((g) => g.split(",").filter(Boolean).map(Number));
      console.log(JSON.stringify(groups.map((g) => decodeIds(g, model)), null, 2));
      break;
    }
    case "trace": {
      const model = store.getModel(Number(opt("version")));
      const texts = symbolTexts(model);
      const t = traceWord(opt("word") ?? "", model);
      console.log(JSON.stringify({ ...t, initialText: t.initial.map((i) => texts[i]), finalText: t.final.map((i) => texts[i]) }, null, 2));
      break;
    }
    case "compare": {
      const m1 = store.getModel(Number(opt("v1"))), m2 = store.getModel(Number(opt("v2")));
      const words = (opt("words") ?? "").split(",");
      console.log(JSON.stringify(words.map((w) => ({ word: w, v1: encodeWord(w, m1), v2: encodeWord(w, m2) })), null, 2));
      break;
    }
    case "export": {
      const data = JSON.stringify(store.exportModel(Number(opt("version"))), null, 2);
      const out = opt("out");
      if (out) { require("fs").writeFileSync(out, data); console.log(`已导出到 ${out}`); }
      else console.log(data);
      break;
    }
    case "corpus": console.log(JSON.stringify(store.getCorpus(Number(opt("version"))), null, 2)); break;
    case "serve": {
      const { createServer } = require("./server");
      const server = createServer(store);
      const port = Number(opt("port", "0"));
      server.listen(port, () => console.log(`BPE API: http://127.0.0.1:${server.address().port}`));
      break;
    }
    default: usage();
  }
} catch (e: any) {
  console.error(`错误 [${e.code ?? "ERROR"}]: ${e.message}`);
  process.exitCode = 1;
} finally {
  if (cmd !== "serve") store.close();
}

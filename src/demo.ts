import { Store } from "./store";
import { encodeWord, decodeIds, symbolTexts, traceWord } from "./core";
import * as fs from "node:fs";

const dbPath = "data/demo.db";
  for (const suffix of ["", "-wal", "-shm"]) { try { fs.unlinkSync(dbPath + suffix); } catch {} }
const store = new Store(dbPath);

const words = [
  { word: "low", freq: 5 },
  { word: "lower", freq: 2 },
  { word: "lowest", freq: 6 },
  { word: "aaaa", freq: 3 },
];
console.log("=== 训练语料 ===");
console.log(words.map((w) => `${w.word}:${w.freq}`).join(", "));

const { version, result } = store.trainNewVersion(words, { maxMerges: 20, minCount: 1 });
console.log(`\n=== 版本 ${version} 训练完成: ${result.stopReason}，词表规模 ${result.vocabSize} ===`);

const texts = symbolTexts(store.getModel(version));
for (const r of result.rounds) {
  const [l, rr] = [r.chosen.left, r.chosen.right];
  console.log(`第 ${r.round} 轮: 合并 (${texts[l]}, ${texts[rr]}) -> "${texts[l] + texts[rr]}" 加权计数=${r.count} 实际替换=${r.replacements} 词表=${r.vocabSize}`);
  console.log(`  候选前3: ${r.candidates.slice(0, 3).map((c) => `(${texts[c.left]},${texts[c.right]})=${c.count}`).join(" ")}`);
}

console.log("\n=== aaaa: 重叠计数 vs 非重叠替换 ===");
const t = traceWord("aaaa", store.getModel(version));
console.log(`初始: ${t.initial.map((i) => texts[i]).join(" ")}`);
for (const s of t.steps.filter((s) => s.applied > 0)) console.log(`第 ${s.round} 轮应用 "${s.text}" x${s.applied} -> ${s.result.map((i) => texts[i]).join(" ")}`);

console.log("\n=== 编码/解码 ===");
const model = store.getModel(version);
for (const w of ["low", "lower", "lowest", "aaaa"]) {
  const ids = encodeWord(w, model);
  console.log(`${w} -> [${ids.join(",")}] ("${ids.map((i) => texts[i]).join("|")}") 解码=${decodeIds(ids, model)}`);
}

console.log("\n=== 再训练创建新版本，旧版本不受影响 ===");
const { version: v2 } = store.trainNewVersion([...words, { word: "newest", freq: 4 }], { maxMerges: 20, minCount: 1 });
for (const w of ["low", "newest"]) {
  const enc = (v: number) => {
    try { return `[${encodeWord(w, store.getModel(v)).join(",")}]`; }
    catch (e: any) { return `错误[${e.code}]: ${e.message}`; }
  };
  console.log(`${w}: v${version}=${enc(version)} v${v2}=${enc(v2)}`);
}
store.close();

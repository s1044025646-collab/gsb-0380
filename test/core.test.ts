import { test } from "node:test";
import assert from "node:assert/strict";
import { train, validateCorpus, encodeWord, decodeIds, traceWord, BpeError, WeightedWord } from "../src/core";
import { Store } from "../src/store";
import * as fs from "node:fs";

// 独立核对实现：直接枚举相邻位置计算每轮频数
function bruteForceCounts(words: { ids: number[]; freq: number }[]): Map<string, number> {
  const m = new Map<string, number>();
  for (const { ids, freq } of words)
    for (let i = 0; i + 1 < ids.length; i++) {
      const k = ids[i] + "," + ids[i + 1];
      m.set(k, (m.get(k) ?? 0) + freq);
    }
  return m;
}

function toSeqs(words: WeightedWord[], alphabet: string[]) {
  const id = new Map(alphabet.map((c, i) => [c, i]));
  return words.map(({ word, freq }) => ({ ids: [...word].map((c) => id.get(c)!), freq }));
}

test("中文码点：按码点拆分与排序分配初始 ID", () => {
  const r = train(validateCorpus([{ word: "中文", freq: 1 }, { word: "文中", freq: 1 }]), { maxMerges: 5, minCount: 1 });
  assert.deepEqual(r.alphabet, ["中", "文"]); // U+4E2D < U+6587
  assert.equal(decodeIds(encodeWord("中文", r), r), "中文");
});

test("重复词行：相同原始字符串权重相加", () => {
  const w = validateCorpus([{ word: "ab", freq: 2 }, { word: "ab", freq: 3 }]);
  assert.deepEqual(w, [{ word: "ab", freq: 5 }]);
});

test("频数并列：按 (left,right) ID 字典序选对", () => {
  const r = train(validateCorpus([{ word: "ba", freq: 1 }]), { maxMerges: 1, minCount: 1 });
  // 初始 ID: a=0, b=1；只有一对 (1,0)
  assert.deepEqual(r.rounds[0].chosen, { left: 1, right: 0 });
  const r2 = train(validateCorpus([{ word: "ab", freq: 1 }, { word: "ba", freq: 1 }]), { maxMerges: 1, minCount: 1 });
  // (0,1) 与 (1,0) 各 1 次，选 ID 字典序小的 (0,1)
  assert.deepEqual(r2.rounds[0].chosen, { left: 0, right: 1 });
});

test("aaaa：计数保留重叠(3对)，替换非重叠(2次)", () => {
  const r = train(validateCorpus([{ word: "aaaa", freq: 1 }]), { maxMerges: 2, minCount: 1 });
  assert.equal(r.rounds[0].count, 3);
  assert.equal(r.rounds[0].replacements, 2);
  assert.equal(r.rounds[1].count, 1); // [A,a] 一对
  assert.equal(r.rounds[1].replacements, 1);
  assert.deepEqual(encodeWord("aaaa", r), [r.merges[1].id]); // 最终折叠为单个符号
});

test("单字符词：没有相邻对时正常结束", () => {
  const r = train(validateCorpus([{ word: "x", freq: 3 }]), { maxMerges: 10, minCount: 1 });
  assert.equal(r.rounds.length, 0);
  assert.match(r.stopReason, /没有相邻符号对/);
});

test("maxMerges=0：不训练任何轮", () => {
  const r = train(validateCorpus([{ word: "ab", freq: 1 }]), { maxMerges: 0, minCount: 1 });
  assert.equal(r.rounds.length, 0);
  assert.match(r.stopReason, /maxMerges/);
});

test("minCount 停止条件", () => {
  const r = train(validateCorpus([{ word: "ab", freq: 2 }]), { maxMerges: 10, minCount: 3 });
  assert.equal(r.rounds.length, 0);
  assert.match(r.stopReason, /minCount/);
});

test("未知字符编码报错，未知 ID 解码报错", () => {
  const r = train(validateCorpus([{ word: "ab", freq: 1 }]), { maxMerges: 1, minCount: 1 });
  assert.throws(() => encodeWord("abc", r), (e: any) => e.code === "UNKNOWN_SYMBOL");
  assert.throws(() => decodeIds([999], r), (e: any) => e.code === "UNKNOWN_ID");
});

test("小语料每轮频数与独立枚举一致", () => {
  const words = validateCorpus([{ word: "low", freq: 5 }, { word: "lower", freq: 2 }, { word: "lowest", freq: 6 }, { word: "aaaa", freq: 3 }]);
  const r = train(words, { maxMerges: 8, minCount: 1 });
  // 重放训练过程，逐轮核对
  let seqs = toSeqs(words, r.alphabet);
  for (const round of r.rounds) {
    const counts = bruteForceCounts(seqs);
    const key = round.chosen.left + "," + round.chosen.right;
    assert.equal(counts.get(key), round.count, `第 ${round.round} 轮选中对计数`);
    // 选中对必须是最大计数
    const max = Math.max(...counts.values());
    assert.equal(round.count, max);
    // 应用替换
    const newId = r.merges[round.round - 1].id;
    let rep = 0;
    for (const s of seqs) {
      const out: number[] = [];
      for (let i = 0; i < s.ids.length; i++) {
        if (i + 1 < s.ids.length && s.ids[i] === round.chosen.left && s.ids[i + 1] === round.chosen.right) { out.push(newId); rep += s.freq; i++; }
        else out.push(s.ids[i]);
      }
      s.ids = out;
    }
    assert.equal(rep, round.replacements, `第 ${round.round} 轮替换次数`);
  }
});

test("解码(编码(词)) === 原词", () => {
  const words = validateCorpus([{ word: "low", freq: 5 }, { word: "lower", freq: 2 }, { word: "lowest", freq: 6 }, { word: "中文测试", freq: 1 }]);
  const r = train(words, { maxMerges: 20, minCount: 1 });
  for (const { word } of words) assert.equal(decodeIds(encodeWord(word, r), r), word);
});

test("重排输入行不改变训练结果", () => {
  const a = train(validateCorpus([{ word: "low", freq: 5 }, { word: "lower", freq: 2 }, { word: "lowest", freq: 6 }]), { maxMerges: 10, minCount: 1 });
  const b = train(validateCorpus([{ word: "lowest", freq: 6 }, { word: "low", freq: 5 }, { word: "lower", freq: 2 }]), { maxMerges: 10, minCount: 1 });
  assert.deepEqual(a.merges, b.merges);
  assert.deepEqual(a.alphabet, b.alphabet);
});

test("编码只按学习优先级，不按最长字符串贪心", () => {
  // 构造：先学 (a,b)->ab，再学 (ab,c)->abc；新词 abc 应按优先级得到 [abc]
  const r = train(validateCorpus([{ word: "abc", freq: 5 }]), { maxMerges: 2, minCount: 1 });
  const ids = encodeWord("abc", r);
  assert.equal(ids.length, 1);
  assert.equal(decodeIds(ids, r), "abc");
});

function tmpDb() {
  const p = `data/test-${Math.random().toString(36).slice(2)}.db`;
  return p;
}

test("SQLite：训练入库、重启后仅凭模型编码、旧版本不受影响", () => {
  const p = tmpDb();
  try {
    let store = new Store(p);
    const { version: v1 } = store.trainNewVersion([{ word: "low", freq: 5 }, { word: "lower", freq: 2 }], { maxMerges: 10, minCount: 1 });
    const enc1 = encodeWord("low", store.getModel(v1));
    store.close();
    // 模拟重启
    store = new Store(p);
    assert.deepEqual(encodeWord("low", store.getModel(v1)), enc1);
    const { version: v2 } = store.trainNewVersion([{ word: "low", freq: 5 }, { word: "lower", freq: 2 }, { word: "lowest", freq: 6 }], { maxMerges: 10, minCount: 1 });
    assert.notEqual(v1, v2);
    assert.deepEqual(encodeWord("low", store.getModel(v1)), enc1); // 旧版本不变
    assert.equal(store.listVersions().length, 2);
    // 轮次日志分页不减少轮数
    const all = store.getRounds(v2, 1, 200);
    const paged = [1, 2, 3].flatMap((pg) => store.getRounds(v2, pg, Math.ceil(all.total / 3)).rounds);
    assert.ok(all.total > 0);
    assert.deepEqual(paged.map((r) => r.round), all.rounds.map((r) => r.round));
    store.close();
  } finally {
    for (const s of ["", "-wal", "-shm"]) try { fs.unlinkSync(p + s); } catch {}
  }
});

test("训练失败回滚：不留下半份模型", () => {
  const p = tmpDb();
  try {
    const store = new Store(p);
    assert.throws(() => store.trainNewVersion([{ word: "ok", freq: 1 }, { word: "bad word", freq: 1 }], { maxMerges: 5, minCount: 1 }), (e: any) => e.code === "INVALID_INPUT");
    assert.equal(store.listVersions().length, 0);
    store.close();
  } finally {
    for (const s of ["", "-wal", "-shm"]) try { fs.unlinkSync(p + s); } catch {}
  }
});

test("参数校验：空白、非正整数频数、超限", () => {
  assert.throws(() => validateCorpus([{ word: "a b", freq: 1 }]), (e: any) => e.code === "INVALID_INPUT");
  assert.throws(() => validateCorpus([{ word: "a", freq: 0 }]), (e: any) => e.code === "INVALID_INPUT");
  assert.throws(() => validateCorpus([{ word: "a".repeat(300), freq: 1 }]), (e: any) => e.code === "LIMIT_EXCEEDED");
  assert.throws(() => validateCorpus([{ word: "a", freq: 2_000_000 }]), (e: any) => e.code === "LIMIT_EXCEEDED");
});

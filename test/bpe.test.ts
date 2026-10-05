import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {
  BpeError,
  Encoder,
  mergeDuplicateWords,
  train,
  WordEntry,
} from '../src/bpe';
import { Store } from '../src/db';
import { startServer } from '../src/server';

// 独立核对实现：直接枚举相邻位置计算每轮频数（与 src/bpe.ts 不共享代码）。
function referenceRoundCounts(
  seqs: { ids: number[]; weight: number }[]
): Map<string, number> {
  const counts = new Map<string, number>();
  for (const seq of seqs) {
    for (let i = 0; i + 1 < seq.ids.length; i++) {
      const key = seq.ids[i] + ',' + seq.ids[i + 1];
      counts.set(key, (counts.get(key) ?? 0) + seq.weight);
    }
  }
  return counts;
}

function referenceTrainSeqs(words: WordEntry[], maxMerges: number) {
  // 独立参考实现：自行分配码点 ID、逐轮枚举计数、选最优、非重叠替换。
  const merged = mergeDuplicateWords(words);
  const cps = [...new Set(merged.flatMap((w) => [...w.word].map((c) => c.codePointAt(0)!)))].sort(
    (a, b) => a - b
  );
  const idOf = new Map(cps.map((cp, i) => [cp, i]));
  let nextId = cps.length;
  let seqs = merged.map((w) => ({
    ids: [...w.word].map((c) => idOf.get(c.codePointAt(0)!)!),
    weight: w.weight,
  }));
  const rounds: { leftId: number; rightId: number; count: number; replacements: number }[] = [];
  for (let r = 0; r < maxMerges; r++) {
    const counts = referenceRoundCounts(seqs);
    if (counts.size === 0) break;
    let best: { l: number; rr: number; c: number } | null = null;
    for (const [key, c] of counts) {
      const [l, rr] = key.split(',').map(Number);
      if (
        !best ||
        c > best.c ||
        (c === best.c && (l < best.l || (l === best.l && rr < best.rr)))
      ) {
        best = { l, rr, c };
      }
    }
    const newId = nextId++;
    let replacements = 0;
    seqs = seqs.map((seq) => {
      const out: number[] = [];
      let i = 0;
      while (i < seq.ids.length) {
        if (i + 1 < seq.ids.length && seq.ids[i] === best!.l && seq.ids[i + 1] === best!.rr) {
          out.push(newId);
          replacements += seq.weight;
          i += 2;
        } else {
          out.push(seq.ids[i]);
          i += 1;
        }
      }
      return { ids: out, weight: seq.weight };
    });
    rounds.push({ leftId: best!.l, rightId: best!.rr, count: best!.c, replacements });
  }
  return rounds;
}

function tmpDb(): { store: Store; cleanup: () => void } {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'bpe-test-'));
  const store = new Store(path.join(dir, 'test.db'));
  return { store, cleanup: () => { store.close(); fs.rmSync(dir, { recursive: true, force: true }); } };
}

test('aaaa: 重叠计数 3，非重叠替换 2', () => {
  const r = train([{ word: 'aaaa', weight: 1 }], 3, 1);
  assert.equal(r.merges[0].count, 3);
  assert.equal(r.merges[0].replacements, 2);
  assert.equal(r.merges[0].text, 'aa');
  assert.equal(r.merges[1].count, 1); // aa aa -> 一对
  assert.equal(r.stopReason, 'no_pairs');
});

test('与独立参考实现逐轮核对频数与替换次数', () => {
  const corpus: WordEntry[] = [
    { word: 'low', weight: 5 },
    { word: 'lower', weight: 2 },
    { word: 'lowest', weight: 3 },
    { word: 'aaaa', weight: 4 },
    { word: 'ababa', weight: 2 },
  ];
  const r = train(corpus, 8, 1);
  const ref = referenceTrainSeqs(corpus, 8);
  assert.equal(r.merges.length, ref.length);
  for (let i = 0; i < ref.length; i++) {
    assert.equal(r.merges[i].leftId, ref[i].leftId, `round ${i + 1} left`);
    assert.equal(r.merges[i].rightId, ref[i].rightId, `round ${i + 1} right`);
    assert.equal(r.merges[i].count, ref[i].count, `round ${i + 1} count`);
    assert.equal(r.merges[i].replacements, ref[i].replacements, `round ${i + 1} replacements`);
  }
});

test('并列时按 (leftId, rightId) 字典序选择', () => {
  // ab 与 cd 各出现 1 次，码点 ID: a=0 b=1 c=2 d=3，应选 (0,1)
  const r = train([{ word: 'ab', weight: 1 }, { word: 'cd', weight: 1 }], 1, 1);
  assert.equal(r.merges[0].leftId, 0);
  assert.equal(r.merges[0].rightId, 1);
  assert.equal(r.merges[0].text, 'ab');
});

test('中文码点：按码点排序分配初始 ID', () => {
  const r = train([{ word: '中中', weight: 2 }, { word: '文中', weight: 1 }], 5, 1);
  const zhong = r.symbols.find((s) => s.text === '中')!;
  const wen = r.symbols.find((s) => s.text === '文')!;
  assert.ok(zhong.id < wen.id); // U+4E2D(中) < U+6587(文)
  assert.equal(r.merges[0].text, '中中'); // 计数 2 最大
});

test('重复词行权重累加，输入行序不影响模型', () => {
  const a: WordEntry[] = [
    { word: 'low', weight: 2 },
    { word: 'lower', weight: 1 },
    { word: 'low', weight: 3 },
  ];
  const b: WordEntry[] = [
    { word: 'lower', weight: 1 },
    { word: 'low', weight: 5 },
  ];
  const ra = train(a, 6, 1);
  const rb = train(b, 6, 1);
  assert.deepEqual(ra.symbols, rb.symbols);
  assert.deepEqual(
    ra.merges.map((m) => [m.leftId, m.rightId, m.count, m.replacements]),
    rb.merges.map((m) => [m.leftId, m.rightId, m.count, m.replacements])
  );
});

test('单字符词：没有相邻对，正常结束 no_pairs', () => {
  const r = train([{ word: 'a', weight: 3 }, { word: 'b', weight: 1 }], 10, 1);
  assert.equal(r.merges.length, 0);
  assert.equal(r.stopReason, 'no_pairs');
});

test('maxMerges 为 0：不训练任何合并', () => {
  const r = train([{ word: 'low', weight: 1 }], 0, 1);
  assert.equal(r.merges.length, 0);
  assert.equal(r.stopReason, 'max_merges');
  assert.equal(r.symbols.length, 3);
});

test('minCount 停止条件', () => {
  const r = train([{ word: 'ab', weight: 1 }, { word: 'abc', weight: 1 }], 10, 3);
  assert.equal(r.merges.length, 0);
  assert.equal(r.stopReason, 'min_count');
});

test('编码/解码往返等于原词；未知字符与未知 ID 报错', () => {
  const r = train(
    [{ word: 'low', weight: 5 }, { word: 'lower', weight: 2 }, { word: 'lowest', weight: 3 }],
    10,
    1
  );
  const enc = new Encoder(r);
  for (const w of ['low', 'lower', 'lowest', 'ow', 'west']) {
    assert.equal(enc.decode(enc.encode(w)), w);
  }
  assert.throws(() => enc.encode('中文'), (e: BpeError) => e.code === 'UNKNOWN_CHAR');
  assert.throws(() => enc.decode([9999]), (e: BpeError) => e.code === 'UNKNOWN_ID');
});

test('编码只按合并优先级，不做最长匹配贪心', () => {
  // 构造：先合并 b+c->bc，再合并 a+bc->abc；词表中不存在 ab。
  const r = train([{ word: 'abc', weight: 1 }, { word: 'bc', weight: 1 }], 5, 1);
  const enc = new Encoder(r);
  const ids = enc.encode('abc');
  const abcSym = r.symbols.find((s) => s.text === 'abc');
  if (abcSym) assert.deepEqual(ids, [abcSym.id]);
  assert.equal(enc.decode(ids), 'abc');
});

test('SQLite：训练后可重启加载编码；修改词频创建新版本且旧版本不变', () => {
  const { store, cleanup } = tmpDb();
  try {
    const m1 = store.importCorpus('m', [{ word: 'low', weight: 5 }, { word: 'lower', weight: 2 }]);
    const r1 = train(store.getCorpus(m1.id), 5, 1);
    store.saveTrainedModel(m1.id, r1, 5, 1);
    const enc1 = new Encoder(store.loadModelData(m1.id));
    const before = enc1.encode('lower');

    const m2 = store.importCorpus('m', [{ word: 'low', weight: 1 }, { word: 'lower', weight: 9 }]);
    assert.equal(m2.version, m1.version + 1);
    const r2 = train(store.getCorpus(m2.id), 5, 1);
    store.saveTrainedModel(m2.id, r2, 5, 1);

    // 旧版本仍按原顺序编码
    const enc1again = new Encoder(store.loadModelData(m1.id));
    assert.deepEqual(enc1again.encode('lower'), before);
    assert.equal(store.loadModelData(m1.id).merges.length, r1.merges.length);
  } finally {
    cleanup();
  }
});

test('SQLite：训练失败不留下可用半份模型', () => {
  const { store, cleanup } = tmpDb();
  try {
    const m = store.importCorpus('m', [{ word: 'low', weight: 1 }]);
    store.markFailed(m.id);
    assert.throws(() => store.loadModelData(m.id), (e: BpeError) => e.code === 'NOT_TRAINED');
  } finally {
    cleanup();
  }
});

test('轮次日志分页不减少实际训练轮数', () => {
  const { store, cleanup } = tmpDb();
  try {
    const m = store.importCorpus('m', [
      { word: 'low', weight: 5 },
      { word: 'lower', weight: 2 },
      { word: 'lowest', weight: 3 },
    ]);
    const r = train(store.getCorpus(m.id), 10, 1);
    store.saveTrainedModel(m.id, r, 10, 1);
    const page1 = store.listRounds(m.id, 1, 2);
    assert.equal(page1.total, r.merges.length);
    assert.equal(page1.rounds.length, Math.min(2, r.merges.length));
    const round1 = store.getRound(m.id, 1);
    assert.equal(round1.round, 1);
    assert.ok(round1.top.length > 0);
  } finally {
    cleanup();
  }
});

test('HTTP API 端到端：导入、训练、编码、解码、轨迹、对比、导出', async () => {
  const { store, cleanup } = tmpDb();
  const { server, port } = await startServer(store);
  const base = `http://127.0.0.1:${port}`;
  try {
    const post = async (p: string, body: unknown) => {
      const res = await fetch(base + p, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      return { status: res.status, json: await res.json() };
    };
    const imp = await post('/corpus/import', {
      name: 'api',
      text: 'low 5\nlower 2\nlowest 3',
    });
    assert.equal(imp.status, 201);
    const id = imp.json.model.id;

    const tr = await post(`/models/${id}/train`, { maxMerges: 10, minCount: 1 });
    assert.equal(tr.status, 200);
    assert.ok(tr.json.rounds > 0);

    const encRes = await post(`/models/${id}/encode`, { words: ['lowest', 'low'] });
    assert.equal(encRes.json.encodings.length, 2);
    const ids = encRes.json.encodings[0].ids;

    const decRes = await post(`/models/${id}/decode`, { ids: [ids] });
    assert.deepEqual(decRes.json.words, ['lowest']);

    const traceRes = await fetch(`${base}/models/${id}/trace?word=lowest`);
    const trace = await traceRes.json();
    assert.ok(trace.steps.length > 1);

    const bad = await post(`/models/${id}/encode`, { words: ['xyz'] });
    assert.equal(bad.status, 400);
    assert.equal(bad.json.error.code, 'UNKNOWN_CHAR');

    const badId = await post(`/models/${id}/decode`, { ids: [[9999]] });
    assert.equal(badId.json.error.code, 'UNKNOWN_ID');

    const exp = await fetch(`${base}/models/${id}/export`);
    const exported = await exp.json();
    assert.ok(exported.symbols.length > 0);

    // 第二版本 + 对比
    const imp2 = await post('/corpus/import', { name: 'api', text: 'low 1\nlower 9' });
    await post(`/models/${imp2.json.model.id}/train`, { maxMerges: 10, minCount: 1 });
    const cmp = await post('/compare', { a: id, b: imp2.json.model.id, words: ['low'] });
    assert.equal(cmp.json.comparisons.length, 1);
    assert.deepEqual(cmp.json.comparisons[0].a.ids, encRes.json.encodings[1].ids);

    const roundsRes = await fetch(`${base}/models/${id}/rounds?page=1&pageSize=2`);
    const rounds = await roundsRes.json();
    assert.equal(rounds.total, tr.json.rounds);
  } finally {
    server.close();
    cleanup();
  }
});

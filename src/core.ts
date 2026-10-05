export interface WeightedWord { word: string; freq: number; }

export interface MergeRecord {
  id: number;
  left: number;
  right: number;
  round: number;
  count: number;        // 加权计数（保留重叠位置）
  replacements: number; // 实际替换次数（从左到右非重叠）
  text: string;         // 展开文本
}

export interface Candidate { left: number; right: number; count: number; }

export interface RoundLog {
  round: number;
  chosen: { left: number; right: number };
  count: number;
  replacements: number;
  vocabSize: number;
  candidates: Candidate[]; // 候选排名摘要（按计数降序、ID 字典序升序）
}

export interface TrainOptions {
  maxMerges: number;
  minCount: number;
  topK?: number; // 每轮日志保留的候选摘要条数，默认 10
}

export interface TrainResult {
  alphabet: string[];            // id -> 码点文本（初始符号）
  merges: MergeRecord[];
  rounds: RoundLog[];
  stopReason: string;
  vocabSize: number;
  totalWeight: number;
}

export class BpeError extends Error {
  constructor(public code: string, message: string) {
    super(message);
    this.name = "BpeError";
  }
}

export const LIMITS = {
  MAX_WORDS: 100_000,
  MAX_WORD_LEN: 256,        // 码点数
  MAX_WEIGHT: 1_000_000,
  MAX_MERGES: 10_000,
  MAX_TOTAL_WEIGHT: 2 ** 40, // 保证加权计数在精确整数范围内
};

export function validateCorpus(input: unknown): WeightedWord[] {
  if (!Array.isArray(input)) throw new BpeError("INVALID_INPUT", "words 必须是数组");
  if (input.length === 0) throw new BpeError("INVALID_INPUT", "语料不能为空");
  if (input.length > LIMITS.MAX_WORDS) throw new BpeError("LIMIT_EXCEEDED", `词条数超过上限 ${LIMITS.MAX_WORDS}`);
  const merged = new Map<string, number>();
  for (const item of input) {
    const w = (item as any)?.word;
    const f = (item as any)?.freq;
    if (typeof w !== "string" || w.length === 0) throw new BpeError("INVALID_INPUT", "word 必须是非空字符串");
    if (/\s/u.test(w)) throw new BpeError("INVALID_INPUT", `单词含空白字符: ${JSON.stringify(w)}`);
    const cps = [...w];
    if (cps.length > LIMITS.MAX_WORD_LEN) throw new BpeError("LIMIT_EXCEEDED", `单词长度超过 ${LIMITS.MAX_WORD_LEN} 码点`);
    if (!Number.isInteger(f) || f < 1) throw new BpeError("INVALID_INPUT", `freq 必须是正整数: ${JSON.stringify(f)}`);
    if (f > LIMITS.MAX_WEIGHT) throw new BpeError("LIMIT_EXCEEDED", `freq 超过上限 ${LIMITS.MAX_WEIGHT}`);
    // 固定规则：相同原始字符串的权重直接相加
    merged.set(w, (merged.get(w) ?? 0) + f);
  }
  let total = 0;
  for (const f of merged.values()) {
    total += f;
    if (total > LIMITS.MAX_TOTAL_WEIGHT) throw new BpeError("LIMIT_EXCEEDED", "加权总数超过精确整数上限");
  }
  return [...merged.entries()].map(([word, freq]) => ({ word, freq }));
}

export function train(words: WeightedWord[], opts: TrainOptions): TrainResult {
  if (!Number.isInteger(opts.maxMerges) || opts.maxMerges < 0) throw new BpeError("INVALID_INPUT", "maxMerges 必须是非负整数");
  if (opts.maxMerges > LIMITS.MAX_MERGES) throw new BpeError("LIMIT_EXCEEDED", `maxMerges 超过上限 ${LIMITS.MAX_MERGES}`);
  if (!Number.isInteger(opts.minCount) || opts.minCount < 1) throw new BpeError("INVALID_INPUT", "minCount 必须是正整数");
  const topK = opts.topK ?? 10;

  // 初始符号：按码点排序分配 ID，输入行顺序不影响结果
  const charSet = new Set<string>();
  for (const { word } of words) for (const ch of word) charSet.add(ch);
  const alphabet = [...charSet].sort((a, b) => a.codePointAt(0)! - b.codePointAt(0)!);
  const charId = new Map(alphabet.map((c, i) => [c, i]));
  const texts: string[] = [...alphabet];

  let seqs = words.map(({ word, freq }) => ({ ids: [...word].map((c) => charId.get(c)!), freq }));
  const merges: MergeRecord[] = [];
  const rounds: RoundLog[] = [];
  let stopReason = "达到最大合并次数 maxMerges";

  for (let round = 1; round <= opts.maxMerges; round++) {
    // 计数阶段：保留重叠位置
    const counts = new Map<string, { left: number; right: number; count: number }>();
    for (const { ids, freq } of seqs) {
      for (let i = 0; i + 1 < ids.length; i++) {
        const key = ids[i] + "," + ids[i + 1];
        const e = counts.get(key);
        if (e) e.count += freq;
        else counts.set(key, { left: ids[i], right: ids[i + 1], count: freq });
      }
    }
    if (counts.size === 0) { stopReason = "当前语料没有相邻符号对"; break; }
    const ranked = [...counts.values()].sort((a, b) => b.count - a.count || a.left - b.left || a.right - b.right);
    const best = ranked[0];
    if (best.count < opts.minCount) { stopReason = `最高对计数 ${best.count} 低于 minCount ${opts.minCount}`; break; }

    // 替换阶段：每个词内从左到右非重叠替换
    const newId = texts.length;
    let replacements = 0;
    for (const s of seqs) {
      const out: number[] = [];
      for (let i = 0; i < s.ids.length; i++) {
        if (i + 1 < s.ids.length && s.ids[i] === best.left && s.ids[i + 1] === best.right) {
          out.push(newId);
          replacements += s.freq;
          i++;
        } else out.push(s.ids[i]);
      }
      s.ids = out;
    }
    const text = texts[best.left] + texts[best.right];
    texts.push(text);
    merges.push({ id: newId, left: best.left, right: best.right, round, count: best.count, replacements, text });
    rounds.push({ round, chosen: { left: best.left, right: best.right }, count: best.count, replacements, vocabSize: texts.length, candidates: ranked.slice(0, topK) });
  }

  let totalWeight = 0;
  for (const { freq } of words) totalWeight += freq;
  return { alphabet, merges, rounds, stopReason, vocabSize: texts.length, totalWeight };
}

export interface Model {
  alphabet: string[];
  merges: MergeRecord[];
}

export function buildCharMap(model: Model): Map<string, number> {
  return new Map(model.alphabet.map((c, i) => [c, i]));
}

export function symbolTexts(model: Model): string[] {
  const texts = [...model.alphabet];
  for (const m of model.merges) texts.push(texts[m.left] + texts[m.right]);
  return texts;
}

export function encodeWord(word: string, model: Model): number[] {
  const charId = buildCharMap(model);
  const ids: number[] = [];
  for (const ch of word) {
    const id = charId.get(ch);
    if (id === undefined) throw new BpeError("UNKNOWN_SYMBOL", `未知码点 U+${ch.codePointAt(0)!.toString(16).toUpperCase().padStart(4, "0")} (${JSON.stringify(ch)})，本模型不支持字节级后备`);
    ids.push(id);
  }
  // 只按已学习的合并优先级依次应用，不做最长匹配贪心
  for (const m of model.merges) {
    const out: number[] = [];
    for (let i = 0; i < ids.length; i++) {
      if (i + 1 < ids.length && ids[i] === m.left && ids[i + 1] === m.right) { out.push(m.id); i++; }
      else out.push(ids[i]);
    }
    ids.length = 0;
    ids.push(...out);
  }
  return ids;
}

export function decodeIds(ids: number[], model: Model): string {
  const texts = symbolTexts(model);
  let out = "";
  for (const id of ids) {
    if (!Number.isInteger(id) || id < 0 || id >= texts.length) throw new BpeError("UNKNOWN_ID", `未知符号 ID: ${id}`);
    out += texts[id];
  }
  return out;
}

export interface TraceStep {
  round: number;
  mergeId: number;
  pair: [number, number];
  text: string;
  applied: number; // 本词中该轮替换次数
  result: number[];
}

export function traceWord(word: string, model: Model): { initial: number[]; steps: TraceStep[]; final: number[] } {
  const charId = buildCharMap(model);
  const ids: number[] = [];
  for (const ch of word) {
    const id = charId.get(ch);
    if (id === undefined) throw new BpeError("UNKNOWN_SYMBOL", `未知码点: ${JSON.stringify(ch)}`);
    ids.push(id);
  }
  const initial = [...ids];
  const steps: TraceStep[] = [];
  for (const m of model.merges) {
    const out: number[] = [];
    let applied = 0;
    for (let i = 0; i < ids.length; i++) {
      if (i + 1 < ids.length && ids[i] === m.left && ids[i + 1] === m.right) { out.push(m.id); applied++; i++; }
      else out.push(ids[i]);
    }
    ids.length = 0;
    ids.push(...out);
    steps.push({ round: m.round, mergeId: m.id, pair: [m.left, m.right], text: m.text, applied, result: [...ids] });
  }
  return { initial, steps, final: [...ids] };
}

// BPE 核心算法：训练、编码、解码、轨迹。纯函数，不依赖存储层。

export interface WordEntry {
  word: string;
  weight: number;
}

export interface SymbolRec {
  id: number;
  text: string;
  leftId: number | null;
  rightId: number | null;
  round: number; // 0 表示初始码点符号
}

export interface PairCount {
  leftId: number;
  rightId: number;
  count: number;
}

export interface MergeRecord {
  round: number;
  leftId: number;
  rightId: number;
  newId: number;
  text: string;
  count: number;         // 加权出现次数（计数阶段保留重叠位置）
  replacements: number;  // 实际替换次数（从左到右非重叠，按词频加权）
  vocabSize: number;     // 本轮之后的词表规模
  top: PairCount[];      // 本轮候选排名摘要
}

export type StopReason = 'max_merges' | 'min_count' | 'no_pairs';

export interface TrainResult {
  symbols: SymbolRec[];
  merges: MergeRecord[];
  stopReason: StopReason;
}

export const LIMITS = {
  maxWords: 100_000,
  maxWordLength: 256, // 码点数
  maxWeight: 1_000_000_000,
  maxMerges: 10_000,
  topK: 10,
};

export class BpeError extends Error {
  constructor(public code: string, message: string) {
    super(message);
  }
}

export function validateCorpus(words: WordEntry[]): void {
  if (!Array.isArray(words) || words.length === 0) {
    throw new BpeError('INVALID_PARAM', '语料不能为空');
  }
  if (words.length > LIMITS.maxWords) {
    throw new BpeError('LIMIT_EXCEEDED', `词数超过上限 ${LIMITS.maxWords}`);
  }
  let total = 0;
  for (const w of words) {
    if (typeof w.word !== 'string' || w.word.length === 0) {
      throw new BpeError('INVALID_PARAM', '单词不能为空');
    }
    if (/\s/.test(w.word)) {
      throw new BpeError('INVALID_PARAM', `单词含空白字符: ${JSON.stringify(w.word)}`);
    }
    const cps = [...w.word];
    if (cps.length > LIMITS.maxWordLength) {
      throw new BpeError('LIMIT_EXCEEDED', `单词长度超过上限 ${LIMITS.maxWordLength}`);
    }
    if (!Number.isSafeInteger(w.weight) || w.weight <= 0) {
      throw new BpeError('INVALID_PARAM', `词频必须为正整数: ${w.weight}`);
    }
    if (w.weight > LIMITS.maxWeight) {
      throw new BpeError('LIMIT_EXCEEDED', `词频超过上限 ${LIMITS.maxWeight}`);
    }
    total += w.weight * cps.length;
    if (!Number.isSafeInteger(total)) {
      throw new BpeError('LIMIT_EXCEEDED', '加权总量超过安全整数范围');
    }
  }
}

// 相同原始字符串的权重合并规则：权重累加；输出按键排序，使输入行顺序不影响结果。
export function mergeDuplicateWords(words: WordEntry[]): WordEntry[] {
  const map = new Map<string, number>();
  for (const w of words) {
    map.set(w.word, (map.get(w.word) ?? 0) + w.weight);
  }
  return [...map.entries()]
    .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .map(([word, weight]) => ({ word, weight }));
}

// 比较器：次数降序；并列时按 (leftId, rightId) 字典序升序。
function comparePairs(a: PairCount, b: PairCount): number {
  if (a.count !== b.count) return b.count - a.count;
  if (a.leftId !== b.leftId) return a.leftId - b.leftId;
  return a.rightId - b.rightId;
}

export function countPairs(seqs: { ids: number[]; weight: number }[]): PairCount[] {
  const counts = new Map<string, PairCount>();
  for (const seq of seqs) {
    for (let i = 0; i + 1 < seq.ids.length; i++) {
      const l = seq.ids[i];
      const r = seq.ids[i + 1];
      const key = l + ',' + r;
      const rec = counts.get(key);
      if (rec) rec.count += seq.weight;
      else counts.set(key, { leftId: l, rightId: r, count: seq.weight });
    }
  }
  return [...counts.values()].sort(comparePairs);
}

// 在一个符号序列上从左到右非重叠替换 (leftId,rightId) -> newId。
export function applyMergeToSeq(
  ids: number[],
  leftId: number,
  rightId: number,
  newId: number
): { ids: number[]; replacements: number } {
  const out: number[] = [];
  let replacements = 0;
  let i = 0;
  while (i < ids.length) {
    if (i + 1 < ids.length && ids[i] === leftId && ids[i + 1] === rightId) {
      out.push(newId);
      replacements++;
      i += 2;
    } else {
      out.push(ids[i]);
      i += 1;
    }
  }
  return { ids: out, replacements };
}

export function train(input: WordEntry[], maxMerges: number, minCount: number): TrainResult {
  if (!Number.isInteger(maxMerges) || maxMerges < 0 || maxMerges > LIMITS.maxMerges) {
    throw new BpeError('INVALID_PARAM', `maxMerges 必须为 0..${LIMITS.maxMerges} 的整数`);
  }
  if (!Number.isInteger(minCount) || minCount < 1) {
    throw new BpeError('INVALID_PARAM', 'minCount 必须为正整数');
  }
  const words = mergeDuplicateWords(input);
  validateCorpus(words);

  // 初始符号：按码点排序分配 ID
  const cpSet = new Set<number>();
  for (const w of words) {
    for (const ch of w.word) cpSet.add(ch.codePointAt(0)!);
  }
  const cps = [...cpSet].sort((a, b) => a - b);
  const idOfCp = new Map<number, number>(cps.map((cp, i) => [cp, i]));
  const symbols: SymbolRec[] = cps.map((cp, i) => ({
    id: i,
    text: String.fromCodePoint(cp),
    leftId: null,
    rightId: null,
    round: 0,
  }));

  let seqs = words.map((w) => ({
    ids: [...w.word].map((ch) => idOfCp.get(ch.codePointAt(0)!)!),
    weight: w.weight,
  }));

  const merges: MergeRecord[] = [];
  let nextId = symbols.length;
  let stopReason: StopReason = 'max_merges';

  for (let round = 1; round <= maxMerges; round++) {
    const ranked = countPairs(seqs);
    if (ranked.length === 0) {
      stopReason = 'no_pairs';
      break;
    }
    const best = ranked[0];
    if (best.count < minCount) {
      stopReason = 'min_count';
      break;
    }
    const newId = nextId++;
    const text = symbols[best.leftId].text + symbols[best.rightId].text;
    symbols.push({ id: newId, text, leftId: best.leftId, rightId: best.rightId, round });

    let replacements = 0;
    seqs = seqs.map((seq) => {
      const r = applyMergeToSeq(seq.ids, best.leftId, best.rightId, newId);
      replacements += r.replacements * seq.weight;
      return { ids: r.ids, weight: seq.weight };
    });

    merges.push({
      round,
      leftId: best.leftId,
      rightId: best.rightId,
      newId,
      text,
      count: best.count,
      replacements,
      vocabSize: symbols.length,
      top: ranked.slice(0, LIMITS.topK),
    });
  }

  return { symbols, merges, stopReason };
}

export interface ModelData {
  symbols: SymbolRec[];
  merges: MergeRecord[];
}

export class Encoder {
  private idOfCp = new Map<number, number>();
  private textOfId = new Map<number, string>();

  constructor(private model: ModelData) {
    for (const s of model.symbols) {
      this.textOfId.set(s.id, s.text);
      if (s.round === 0) this.idOfCp.set(s.text.codePointAt(0)!, s.id);
    }
  }

  private initialIds(word: string): number[] {
    if (typeof word !== 'string' || word.length === 0 || /\s/.test(word)) {
      throw new BpeError('INVALID_PARAM', `非法单词: ${JSON.stringify(word)}`);
    }
    return [...word].map((ch) => {
      const cp = ch.codePointAt(0)!;
      const id = this.idOfCp.get(cp);
      if (id === undefined) {
        throw new BpeError(
          'UNKNOWN_CHAR',
          `未知字符 U+${cp.toString(16).toUpperCase().padStart(4, '0')} (${ch})`
        );
      }
      return id;
    });
  }

  // 按学习到的合并优先级（轮次顺序）应用规则；未知初始码点报错。
  encode(word: string): number[] {
    let ids = this.initialIds(word);
    for (const m of this.model.merges) {
      ids = applyMergeToSeq(ids, m.leftId, m.rightId, m.newId).ids;
      if (ids.length <= 1) break;
    }
    return ids;
  }

  // 返回每一步合并后的序列，用于轨迹展示。
  trace(word: string): { round: number; text: string; ids: number[]; applied: boolean }[] {
    let ids = this.initialIds(word);
    const steps: { round: number; text: string; ids: number[]; applied: boolean }[] = [
      { round: 0, text: '初始码点拆分', ids: [...ids], applied: false },
    ];
    for (const m of this.model.merges) {
      const r = applyMergeToSeq(ids, m.leftId, m.rightId, m.newId);
      steps.push({ round: m.round, text: m.text, ids: r.ids, applied: r.replacements > 0 });
      ids = r.ids;
      if (ids.length <= 1) break;
    }
    return steps;
  }

  decode(ids: number[]): string {
    let out = '';
    for (const id of ids) {
      const text = this.textOfId.get(id);
      if (text === undefined) {
        throw new BpeError('UNKNOWN_ID', `未知符号 ID: ${id}`);
      }
      out += text;
    }
    return out;
  }
}

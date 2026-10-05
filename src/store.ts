import { DatabaseSync } from "node:sqlite";
import * as path from "node:path";
import * as fs from "node:fs";
import { BpeError, MergeRecord, Model, RoundLog, TrainOptions, TrainResult, train, validateCorpus, WeightedWord } from "./core";

export const ALGO_VERSION = "bpe-w1";

export interface VersionInfo {
  version: number;
  algo: string;
  createdAt: string;
  maxMerges: number;
  minCount: number;
  stopReason: string;
  vocabSize: number;
  totalWeight: number;
  wordCount: number;
}

export class Store {
  db: DatabaseSync;
  constructor(dbPath: string) {
    fs.mkdirSync(path.dirname(dbPath), { recursive: true });
    this.db = new DatabaseSync(dbPath);
    this.db.exec(`
      PRAGMA journal_mode = WAL;
      CREATE TABLE IF NOT EXISTS versions (
        version INTEGER PRIMARY KEY AUTOINCREMENT,
        algo TEXT NOT NULL,
        created_at TEXT NOT NULL,
        max_merges INTEGER NOT NULL,
        min_count INTEGER NOT NULL,
        stop_reason TEXT NOT NULL,
        vocab_size INTEGER NOT NULL,
        total_weight INTEGER NOT NULL,
        word_count INTEGER NOT NULL
      );
      CREATE TABLE IF NOT EXISTS corpus (
        version INTEGER NOT NULL, word TEXT NOT NULL, freq INTEGER NOT NULL,
        PRIMARY KEY (version, word)
      );
      CREATE TABLE IF NOT EXISTS symbols (
        version INTEGER NOT NULL, id INTEGER NOT NULL, text TEXT NOT NULL,
        left_id INTEGER, right_id INTEGER, round INTEGER,
        PRIMARY KEY (version, id)
      );
      CREATE TABLE IF NOT EXISTS rounds (
        version INTEGER NOT NULL, round INTEGER NOT NULL,
        left_id INTEGER NOT NULL, right_id INTEGER NOT NULL,
        count INTEGER NOT NULL, replacements INTEGER NOT NULL,
        vocab_size INTEGER NOT NULL, candidates TEXT NOT NULL,
        PRIMARY KEY (version, round)
      );
    `);
  }

  // 训练失败整体回滚，不留下半份可用模型
  trainNewVersion(rawWords: unknown, opts: TrainOptions): { version: number; result: TrainResult } {
    const words = validateCorpus(rawWords);
    this.db.exec("BEGIN");
    try {
      const result = train(words, opts);
      const ins = this.db.prepare(
        "INSERT INTO versions (algo, created_at, max_merges, min_count, stop_reason, vocab_size, total_weight, word_count) VALUES (?,?,?,?,?,?,?,?)"
      );
      const info = ins.run(ALGO_VERSION, new Date().toISOString(), opts.maxMerges, opts.minCount, result.stopReason, result.vocabSize, result.totalWeight, words.length);
      const version = Number(info.lastInsertRowid);
      const insCorpus = this.db.prepare("INSERT INTO corpus (version, word, freq) VALUES (?,?,?)");
      for (const w of words) insCorpus.run(version, w.word, w.freq);
      const insSym = this.db.prepare("INSERT INTO symbols (version, id, text, left_id, right_id, round) VALUES (?,?,?,?,?,?)");
      result.alphabet.forEach((text, id) => insSym.run(version, id, text, null, null, null));
      for (const m of result.merges) insSym.run(version, m.id, m.text, m.left, m.right, m.round);
      const insRound = this.db.prepare("INSERT INTO rounds (version, round, left_id, right_id, count, replacements, vocab_size, candidates) VALUES (?,?,?,?,?,?,?,?)");
      for (const r of result.rounds) insRound.run(version, r.round, r.chosen.left, r.chosen.right, r.count, r.replacements, r.vocabSize, JSON.stringify(r.candidates));
      this.db.exec("COMMIT");
      return { version, result };
    } catch (e) {
      this.db.exec("ROLLBACK");
      throw e;
    }
  }

  listVersions(): VersionInfo[] {
    const rows = this.db.prepare("SELECT * FROM versions ORDER BY version").all() as any[];
    return rows.map(mapVersion);
  }

  getVersion(version: number): VersionInfo {
    const row = this.db.prepare("SELECT * FROM versions WHERE version = ?").get(version) as any;
    if (!row) throw new BpeError("VERSION_NOT_FOUND", `版本 ${version} 不存在`);
    return mapVersion(row);
  }

  getCorpus(version: number): WeightedWord[] {
    this.getVersion(version);
    const rows = this.db.prepare("SELECT word, freq FROM corpus WHERE version = ? ORDER BY word").all(version) as any[];
    return rows.map((r) => ({ word: r.word, freq: r.freq }));
  }

  // 仅凭保存的模型即可重建编码能力（重启后无需语料）
  getModel(version: number): Model {
    this.getVersion(version);
    const rows = this.db.prepare("SELECT id, text, left_id, right_id, round FROM symbols WHERE version = ? ORDER BY id").all(version) as any[];
    const alphabet: string[] = [];
    const merges: MergeRecord[] = [];
    for (const r of rows) {
      if (r.left_id === null) alphabet[r.id] = r.text;
      else merges.push({ id: r.id, left: r.left_id, right: r.right_id, round: r.round, count: 0, replacements: 0, text: r.text });
    }
    return { alphabet, merges };
  }

  getRounds(version: number, page: number, pageSize: number): { total: number; rounds: RoundLog[] } {
    this.getVersion(version);
    const total = (this.db.prepare("SELECT COUNT(*) c FROM rounds WHERE version = ?").get(version) as any).c as number;
    const rows = this.db.prepare("SELECT * FROM rounds WHERE version = ? ORDER BY round LIMIT ? OFFSET ?").all(version, pageSize, (page - 1) * pageSize) as any[];
    return { total, rounds: rows.map(mapRound) };
  }

  getRound(version: number, round: number): RoundLog {
    const row = this.db.prepare("SELECT * FROM rounds WHERE version = ? AND round = ?").get(version, round) as any;
    if (!row) throw new BpeError("ROUND_NOT_FOUND", `版本 ${version} 没有第 ${round} 轮`);
    return mapRound(row);
  }

  exportModel(version: number) {
    const info = this.getVersion(version);
    const model = this.getModel(version);
    const rounds: RoundLog[] = [];
    let page = 1;
    for (;;) {
      const { total, rounds: rs } = this.getRounds(version, page, 500);
      rounds.push(...rs);
      if (rounds.length >= total) break;
      page++;
    }
    return { algo: info.algo, version: info.version, params: { maxMerges: info.maxMerges, minCount: info.minCount }, stopReason: info.stopReason, vocabSize: info.vocabSize, totalWeight: info.totalWeight, alphabet: model.alphabet, merges: model.merges, rounds };
  }

  close() { this.db.close(); }
}

function mapVersion(r: any): VersionInfo {
  return { version: r.version, algo: r.algo, createdAt: r.created_at, maxMerges: r.max_merges, minCount: r.min_count, stopReason: r.stop_reason, vocabSize: r.vocab_size, totalWeight: r.total_weight, wordCount: r.word_count };
}
function mapRound(r: any): RoundLog {
  return { round: r.round, chosen: { left: r.left_id, right: r.right_id }, count: r.count, replacements: r.replacements, vocabSize: r.vocab_size, candidates: JSON.parse(r.candidates) };
}

export function defaultDbPath(): string {
  return path.join(process.cwd(), "data", "bpe.db");
}

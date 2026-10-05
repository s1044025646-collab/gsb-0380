// SQLite 持久层：模型版本、加权语料、符号表、合并规则与轮次日志。
import { DatabaseSync } from 'node:sqlite';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { BpeError, MergeRecord, ModelData, SymbolRec, TrainResult, WordEntry } from './bpe';

export const ALGO_VERSION = 'bpe-word-v1';

export interface ModelRow {
  id: number;
  name: string;
  version: number;
  status: string;
  algoVersion: string;
  stopReason: string | null;
  maxMerges: number | null;
  minCount: number | null;
  vocabSize: number;
  createdAt: string;
}

export class Store {
  private db: DatabaseSync;

  constructor(dbPath: string) {
    fs.mkdirSync(path.dirname(dbPath), { recursive: true });
    this.db = new DatabaseSync(dbPath);
    this.db.exec(`
      PRAGMA journal_mode = WAL;
      CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        version INTEGER NOT NULL,
        status TEXT NOT NULL,            -- training | ready | failed
        algo_version TEXT NOT NULL,
        stop_reason TEXT,
        max_merges INTEGER,
        min_count INTEGER,
        vocab_size INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        UNIQUE(name, version)
      );
      CREATE TABLE IF NOT EXISTS corpus (
        model_id INTEGER NOT NULL REFERENCES models(id),
        word TEXT NOT NULL,
        weight INTEGER NOT NULL,
        PRIMARY KEY (model_id, word)
      );
      CREATE TABLE IF NOT EXISTS symbols (
        model_id INTEGER NOT NULL REFERENCES models(id),
        id INTEGER NOT NULL,
        text TEXT NOT NULL,
        left_id INTEGER,
        right_id INTEGER,
        round INTEGER NOT NULL,
        PRIMARY KEY (model_id, id)
      );
      CREATE TABLE IF NOT EXISTS merges (
        model_id INTEGER NOT NULL REFERENCES models(id),
        round INTEGER NOT NULL,
        left_id INTEGER NOT NULL,
        right_id INTEGER NOT NULL,
        new_id INTEGER NOT NULL,
        text TEXT NOT NULL,
        count INTEGER NOT NULL,
        replacements INTEGER NOT NULL,
        vocab_size INTEGER NOT NULL,
        top_json TEXT NOT NULL,
        PRIMARY KEY (model_id, round)
      );
    `);
  }

  close(): void {
    this.db.close();
  }

  private now(): string {
    return new Date().toISOString();
  }

  /** 导入语料：同名模型创建新版本；训练前状态为 training。 */
  importCorpus(name: string, words: WordEntry[]): ModelRow {
    if (!name || typeof name !== 'string') {
      throw new BpeError('INVALID_PARAM', '模型名不能为空');
    }
    const row = this.db
      .prepare('SELECT MAX(version) AS v FROM models WHERE name = ?')
      .get(name) as { v: number | null };
    const version = (row.v ?? 0) + 1;
    const info = this.db
      .prepare(
        `INSERT INTO models (name, version, status, algo_version, created_at)
         VALUES (?, ?, 'training', ?, ?)`
      )
      .run(name, version, ALGO_VERSION, this.now());
    const modelId = Number(info.lastInsertRowid);
    const ins = this.db.prepare('INSERT INTO corpus (model_id, word, weight) VALUES (?, ?, ?)');
    for (const w of words) ins.run(modelId, w.word, w.weight);
    return this.getModel(modelId);
  }

  getCorpus(modelId: number): WordEntry[] {
    const rows = this.db
      .prepare('SELECT word, weight FROM corpus WHERE model_id = ? ORDER BY word')
      .all(modelId) as unknown as { word: string; weight: number }[];
    return rows.map((r) => ({ word: r.word, weight: r.weight }));
  }

  /** 训练结果落库：整体事务，失败回滚并标记 failed，不留下可用半份模型。 */
  saveTrainedModel(modelId: number, result: TrainResult, maxMerges: number, minCount: number): void {
    this.db.exec('BEGIN');
    try {
      const insSym = this.db.prepare(
        'INSERT INTO symbols (model_id, id, text, left_id, right_id, round) VALUES (?, ?, ?, ?, ?, ?)'
      );
      for (const s of result.symbols) {
        insSym.run(modelId, s.id, s.text, s.leftId, s.rightId, s.round);
      }
      const insMerge = this.db.prepare(
        `INSERT INTO merges (model_id, round, left_id, right_id, new_id, text, count, replacements, vocab_size, top_json)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      );
      for (const m of result.merges) {
        insMerge.run(
          modelId, m.round, m.leftId, m.rightId, m.newId, m.text,
          m.count, m.replacements, m.vocabSize, JSON.stringify(m.top)
        );
      }
      this.db
        .prepare(
          `UPDATE models SET status = 'ready', stop_reason = ?, max_merges = ?, min_count = ?, vocab_size = ?
           WHERE id = ?`
        )
        .run(result.stopReason, maxMerges, minCount, result.symbols.length, modelId);
      this.db.exec('COMMIT');
    } catch (err) {
      this.db.exec('ROLLBACK');
      this.db.prepare("UPDATE models SET status = 'failed' WHERE id = ?").run(modelId);
      throw err;
    }
  }

  markFailed(modelId: number): void {
    this.db.prepare("UPDATE models SET status = 'failed' WHERE id = ?").run(modelId);
  }

  private toModelRow(r: any): ModelRow {
    return {
      id: r.id,
      name: r.name,
      version: r.version,
      status: r.status,
      algoVersion: r.algo_version,
      stopReason: r.stop_reason,
      maxMerges: r.max_merges,
      minCount: r.min_count,
      vocabSize: r.vocab_size,
      createdAt: r.created_at,
    };
  }

  getModel(id: number): ModelRow {
    const r = this.db.prepare('SELECT * FROM models WHERE id = ?').get(id);
    if (!r) throw new BpeError('MODEL_NOT_FOUND', `模型不存在: ${id}`);
    return this.toModelRow(r);
  }

  listModels(): ModelRow[] {
    const rows = this.db.prepare('SELECT * FROM models ORDER BY name, version').all();
    return (rows as any[]).map((r) => this.toModelRow(r));
  }

  loadModelData(modelId: number): ModelData {
    const model = this.getModel(modelId);
    if (model.status !== 'ready') {
      throw new BpeError('NOT_TRAINED', `模型 ${modelId} 不可用（状态: ${model.status}）`);
    }
    const symbols = (
      this.db
        .prepare('SELECT id, text, left_id, right_id, round FROM symbols WHERE model_id = ? ORDER BY id')
        .all(modelId) as any[]
    ).map(
      (r): SymbolRec => ({ id: r.id, text: r.text, leftId: r.left_id, rightId: r.right_id, round: r.round })
    );
    const merges = (
      this.db
        .prepare('SELECT * FROM merges WHERE model_id = ? ORDER BY round')
        .all(modelId) as any[]
    ).map(
      (r): MergeRecord => ({
        round: r.round,
        leftId: r.left_id,
        rightId: r.right_id,
        newId: r.new_id,
        text: r.text,
        count: r.count,
        replacements: r.replacements,
        vocabSize: r.vocab_size,
        top: JSON.parse(r.top_json),
      })
    );
    return { symbols, merges };
  }

  getRound(modelId: number, round: number): MergeRecord {
    const r = this.db
      .prepare('SELECT * FROM merges WHERE model_id = ? AND round = ?')
      .get(modelId, round) as any;
    if (!r) throw new BpeError('ROUND_NOT_FOUND', `模型 ${modelId} 没有第 ${round} 轮`);
    return {
      round: r.round,
      leftId: r.left_id,
      rightId: r.right_id,
      newId: r.new_id,
      text: r.text,
      count: r.count,
      replacements: r.replacements,
      vocabSize: r.vocab_size,
      top: JSON.parse(r.top_json),
    };
  }

  listRounds(modelId: number, page: number, pageSize: number): { total: number; rounds: MergeRecord[] } {
    const t = this.db
      .prepare('SELECT COUNT(*) AS c FROM merges WHERE model_id = ?')
      .get(modelId) as any;
    const rows = this.db
      .prepare('SELECT * FROM merges WHERE model_id = ? ORDER BY round LIMIT ? OFFSET ?')
      .all(modelId, pageSize, (page - 1) * pageSize) as any[];
    return {
      total: t.c,
      rounds: rows.map((r) => ({
        round: r.round,
        leftId: r.left_id,
        rightId: r.right_id,
        newId: r.new_id,
        text: r.text,
        count: r.count,
        replacements: r.replacements,
        vocabSize: r.vocab_size,
        top: JSON.parse(r.top_json),
      })),
    };
  }

  exportModel(modelId: number): object {
    const model = this.getModel(modelId);
    const data = this.loadModelData(modelId);
    return {
      name: model.name,
      version: model.version,
      algoVersion: model.algoVersion,
      stopReason: model.stopReason,
      params: { maxMerges: model.maxMerges, minCount: model.minCount },
      symbols: data.symbols,
      merges: data.merges,
    };
  }
}

export function defaultDbPath(): string {
  return path.join(process.cwd(), 'data', 'bpe.db');
}

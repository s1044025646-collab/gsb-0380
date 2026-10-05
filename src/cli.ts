// CLI：import / train / rounds / round / encode / decode / trace / compare / export / serve / demo
import * as fs from 'node:fs';
import { Encoder, train, WordEntry } from './bpe';
import { defaultDbPath, Store } from './db';
import { startServer } from './server';

function argValue(args: string[], name: string): string | undefined {
  const i = args.indexOf('--' + name);
  return i >= 0 ? args[i + 1] : undefined;
}

function parseCorpusText(text: string): WordEntry[] {
  const out: WordEntry[] = [];
  for (const line of text.split(/\r?\n/)) {
    const t = line.trim();
    if (!t) continue;
    const m = t.match(/^(\S+)\s+(\d+)$/);
    if (!m) throw new Error(`语料行格式错误（应为 "单词 正整数词频"）: ${t}`);
    out.push({ word: m[1], weight: Number(m[2]) });
  }
  return out;
}

function print(obj: unknown): void {
  console.log(JSON.stringify(obj, null, 2));
}

const HELP = [
  'BPE 子词训练后端 CLI',
  '',
  '用法: node dist/cli.js <命令> [参数]',
  '',
  '命令:',
  '  import  --name N (--file F | --text "low 5")   导入语料，创建新版本',
  '  train   --model ID [--max-merges 100] [--min-count 1]   训练模型',
  '  models                                          列出模型',
  '  rounds  --model ID [--page 1] [--page-size 20]  分页查看轮次日志',
  '  round   --model ID --round N                    查看某一轮详情',
  '  encode  --model ID --words "low,lower"          编码词列表',
  '  decode  --model ID --ids "[[1,2],[3]]"          解码 ID 列表',
  '  trace   --model ID --word lowest                查看单词合并轨迹',
  '  compare --a ID --b ID --words "low,lower"       两个版本编码对照',
  '  export  --model ID [--out model.json]           导出模型 JSON',
  '  serve   [--port 3000]                           启动 HTTP API（默认选空闲端口）',
  '  demo                                            运行内置演示',
  '',
  '全局: --db PATH 指定数据库文件（默认 ./data/bpe.db）',
  '',
].join('\n');

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const cmd = args[0];
  if (!cmd || cmd === 'help' || cmd === '--help') {
    console.log(HELP);
    return;
  }
  const store = new Store(argValue(args, 'db') ?? defaultDbPath());
  try {
    switch (cmd) {
      case 'import': {
        const name = argValue(args, 'name');
        if (!name) throw new Error('缺少 --name');
        const file = argValue(args, 'file');
        const text = argValue(args, 'text');
        if (!file && !text) throw new Error('需要 --file 或 --text');
        const words = parseCorpusText(file ? fs.readFileSync(file, 'utf8') : text!);
        print({ model: store.importCorpus(name, words) });
        break;
      }
      case 'train': {
        const id = Number(argValue(args, 'model'));
        const maxMerges = Number(argValue(args, 'max-merges') ?? 100);
        const minCount = Number(argValue(args, 'min-count') ?? 1);
        const corpus = store.getCorpus(id);
        try {
          const result = train(corpus, maxMerges, minCount);
          store.saveTrainedModel(id, result, maxMerges, minCount);
          print({ model: store.getModel(id), stopReason: result.stopReason, rounds: result.merges.length });
        } catch (err) {
          store.markFailed(id);
          throw err;
        }
        break;
      }
      case 'models':
        print({ models: store.listModels() });
        break;
      case 'rounds': {
        const id = Number(argValue(args, 'model'));
        const page = Number(argValue(args, 'page') ?? 1);
        const pageSize = Number(argValue(args, 'page-size') ?? 20);
        print(store.listRounds(id, page, pageSize));
        break;
      }
      case 'round':
        print({ round: store.getRound(Number(argValue(args, 'model')), Number(argValue(args, 'round'))) });
        break;
      case 'encode': {
        const enc = new Encoder(store.loadModelData(Number(argValue(args, 'model'))));
        const words = (argValue(args, 'words') ?? '').split(',').filter(Boolean);
        print({ encodings: words.map((w) => ({ word: w, ids: enc.encode(w) })) });
        break;
      }
      case 'decode': {
        const enc = new Encoder(store.loadModelData(Number(argValue(args, 'model'))));
        const ids = JSON.parse(argValue(args, 'ids') ?? '[]') as number[][];
        print({ words: ids.map((row) => enc.decode(row)) });
        break;
      }
      case 'trace': {
        const enc = new Encoder(store.loadModelData(Number(argValue(args, 'model'))));
        print({ steps: enc.trace(argValue(args, 'word') ?? '') });
        break;
      }
      case 'compare': {
        const a = Number(argValue(args, 'a'));
        const b = Number(argValue(args, 'b'));
        const encA = new Encoder(store.loadModelData(a));
        const encB = new Encoder(store.loadModelData(b));
        const words = (argValue(args, 'words') ?? '').split(',').filter(Boolean);
        print({
          comparisons: words.map((w) => ({
            word: w,
            a: { model: a, ids: encA.encode(w) },
            b: { model: b, ids: encB.encode(w) },
          })),
        });
        break;
      }
      case 'export': {
        const data = store.exportModel(Number(argValue(args, 'model')));
        const out = argValue(args, 'out');
        if (out) {
          fs.writeFileSync(out, JSON.stringify(data, null, 2), 'utf8');
          console.log(`已导出到 ${out}`);
        } else {
          print(data);
        }
        break;
      }
      case 'serve': {
        const portArg = argValue(args, 'port');
        const { port } = await startServer(store, portArg ? Number(portArg) : undefined);
        console.log(`BPE API 已启动: http://127.0.0.1:${port}`);
        break;
      }
      case 'demo':
        runDemo(store);
        break;
      default:
        console.error(`未知命令: ${cmd}`);
        console.log(HELP);
        process.exitCode = 1;
    }
  } finally {
    if (args[0] !== 'serve') store.close();
  }
}

function runDemo(store: Store): void {
  console.log('== 演示 1: low / lower / lowest 公共子词逐轮出现 ==');
  const m1 = store.importCorpus('demo-en', [
    { word: 'low', weight: 5 },
    { word: 'lower', weight: 2 },
    { word: 'lowest', weight: 3 },
  ]);
  const r1 = train(store.getCorpus(m1.id), 10, 1);
  store.saveTrainedModel(m1.id, r1, 10, 1);
  for (const m of r1.merges) {
    console.log(
      `第 ${m.round} 轮: 合并 (${m.leftId},${m.rightId}) -> "${m.text}"(id=${m.newId}) ` +
        `加权计数=${m.count} 实际替换=${m.replacements} 词表=${m.vocabSize}`
    );
  }
  console.log(`停止原因: ${r1.stopReason}`);
  const enc1 = new Encoder(store.loadModelData(m1.id));
  console.log('编码 lowest:', enc1.encode('lowest'));
  console.log('轨迹 lowest:', JSON.stringify(enc1.trace('lowest').filter((s) => s.applied || s.round === 0)));

  console.log('');
  console.log('== 演示 2: aaaa 重叠计数 vs 非重叠替换 ==');
  const m2 = store.importCorpus('demo-aaaa', [{ word: 'aaaa', weight: 1 }]);
  const r2 = train(store.getCorpus(m2.id), 3, 1);
  store.saveTrainedModel(m2.id, r2, 3, 1);
  for (const m of r2.merges) {
    console.log(
      `第 ${m.round} 轮: "${m.text}" 加权计数=${m.count}（重叠计入）实际替换=${m.replacements}（非重叠）`
    );
  }
  const enc2 = new Encoder(store.loadModelData(m2.id));
  console.log('编码 aaaa:', enc2.encode('aaaa'), '解码:', enc2.decode(enc2.encode('aaaa')));
}

main().catch((err) => {
  console.error(`错误 [${err.code ?? 'ERROR'}]: ${err.message}`);
  process.exitCode = 1;
});

# BPE 子词词表训练与合并次序解释后端

面向**加权单词清单**的 BPE（Byte-Pair Encoding）子词训练与编解码服务。仅提供 HTTP API 与 CLI，使用 TypeScript + Node.js + SQLite（`node:sqlite` 内置模块），Windows 本地运行，无 Docker / WSL / 外部服务依赖。

## 快速开始

```powershell
npm install        # 安装 devDependencies（typescript、@types/node）
npm run build      # 编译到 dist/
npm test           # 构建并运行全部测试
npm start          # 启动 API（默认自动选择空闲端口）
npm start -- --port=8080   # 指定端口
npm run demo       # 演示：low/lower/lowest + aaaa
node dist/src/cli.js train --words "low:5,lower:2,lowest:6,aaaa:3" --max-merges 20
```

数据保存在项目内 `data/` 目录（默认 `data/bpe.db`）。

## 算法口径（重要）

- **码点口径**：输入单词按 Unicode 码点（非 UTF-16 代码单元、非字素簇）拆分为初始符号。不做大小写折叠，不做 Unicode 归一化（NFC/NFD 视为不同输入）。
- **初始 ID**：所有出现过的码点按码点数值升序分配 ID `0..n-1`，因此**输入行顺序不影响训练结果**。
- **重复词行**：相同原始字符串的频数直接相加（固定规则）。
- **每轮训练**：
  1. **计数阶段**：对当前分词统计相邻符号对的加权出现次数，**保留重叠位置**——`aaaa`（频数 1）有 3 个 `(a,a)` 相邻对。
  2. **选择**：取计数最大者；并列时按组成符号的两个数字 ID `(left, right)` 字典序选取。
  3. **替换阶段**：每个词内**从左到右非重叠**替换——`aaaa` 只替换 2 次。日志同时记录加权计数与实际替换次数，二者不混淆。
- **符号身份**：每个合并产生新 ID，记录左右父符号 ID、展开文本与轮次。ID 是身份，展开文本只是展示；不同合并路径即使文本相同也不歧义。
- **停止条件**（返回真实原因）：达到 `maxMerges`、最高对计数低于 `minCount`、或当前语料没有相邻符号对（如全是单字符词）。
- **训练日志来自真实语料状态**：每轮在真实分词上计算候选排名摘要（默认前 10）、选中的对、加权计数、实际替换次数与词表规模，不是事后补写。

## 编解码语义

- **编码**：新词先映射到初始码点 ID，再**只按已学习的合并优先级（轮次顺序）**依次应用规则；不按最长字符串贪心，不对新输入重新训练。未知码点报 `UNKNOWN_SYMBOL`（本题不做字节级后备）。
- **解码**：按符号展开文本连接还原；未知 ID 报 `UNKNOWN_ID`。
- **词边界**：编码/解码的输入输出都是数组的数组，完整单词与多词边界由外层数组保留，不引入隐含空格或特殊结束符。
- **重启恢复**：模型（字符表 + 合并规则）持久化在 SQLite，重启后仅凭保存模型即可编码新词。
- **版本**：每次训练创建新版本；修改词频再训练生成新版本，旧版本仍按原合并顺序编码。

## HTTP API

默认端口为 0（自动选空闲端口），启动时打印实际端口。所有错误返回 `{"error":{"code","message"}}`。

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/train` | `{words:[{word,freq}], maxMerges?, minCount?}` 训练新版本 |
| GET | `/versions` | 版本列表 |
| GET | `/versions/:v` | 版本信息（含停止原因） |
| GET | `/versions/:v/corpus` | 该版本的加权语料 |
| GET | `/versions/:v/rounds?page=&pageSize=` | 分页轮次日志（分页不减少实际训练轮数） |
| GET | `/versions/:v/rounds/:n` | 第 n 轮详情（候选排名、选中对、计数、替换数、词表规模） |
| POST | `/versions/:v/encode` | `{words:["low",...]}` → `{tokens:[[...],...]}` |
| POST | `/versions/:v/decode` | `{ids:[[...],...]}` → `{words:[...]}` |
| GET | `/versions/:v/trace?word=low` | 指定词的逐轮合并轨迹 |
| POST | `/compare` | `{v1,v2,words:[...]}` 两个版本编码对照 |
| GET | `/versions/:v/export` | 导出完整模型 JSON（字符表、合并、轮次日志） |

错误码：`INVALID_INPUT`、`LIMIT_EXCEEDED`、`UNKNOWN_SYMBOL`、`UNKNOWN_ID`、`VERSION_NOT_FOUND`、`ROUND_NOT_FOUND`、`NOT_FOUND`、`INTERNAL`。

## CLI

```powershell
node dist/src/cli.js train --words "low:5,lower:2" --max-merges 20 --min-count 1
node dist/src/cli.js versions
node dist/src/cli.js rounds --version 1 --page 1 --page-size 20
node dist/src/cli.js round --version 1 --round 3
node dist/src/cli.js encode --version 1 --words "low,lower"
node dist/src/cli.js decode --version 1 --ids "9;14"   # 分号分隔多个词
node dist/src/cli.js trace --version 1 --word aaaa
node dist/src/cli.js compare --v1 1 --v2 2 --words "low,lower"
node dist/src/cli.js export --version 1 --out model.json
node dist/src/cli.js corpus --version 1
node dist/src/cli.js serve --port 8080
```

## 限制与上限

- 只研究**词内**子词构造：不做跨词合并、语言模型推理、词性判断、词干提取或文本压缩。
- 上限（`src/core.ts` 的 `LIMITS`）：词条数 ≤ 100,000；词长 ≤ 256 码点；词频 ≤ 1,000,000；合并轮数 ≤ 10,000；加权总数 ≤ 2^40（保证计数为精确整数）。
- 训练在单事务内完成，失败整体回滚，不会留下半份可用模型。
- `node:sqlite` 目前带 ExperimentalWarning，属正常现象。

## 项目结构

- `src/core.ts` — 纯 BPE 训练/编码/解码/轨迹逻辑与参数校验
- `src/store.ts` — SQLite 持久化（语料、字符表、合并规则、轮次日志、算法版本）
- `src/server.ts` — HTTP API（node:http，无框架）
- `src/cli.ts` — 命令行入口
- `src/demo.ts` — 演示脚本（low/lower/lowest 公共子词 + aaaa 重叠计数）
- `test/core.test.ts` — 15 个测试（中文码点、重复行、并列、aaaa、独立枚举核对、版本隔离、回滚等）

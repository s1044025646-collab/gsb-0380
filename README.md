# BPE 子词词表训练与合并次序解释后端

面向**加权单词清单**的 BPE 子词训练与编解码服务。仅做后端：HTTP API + CLI，
使用 TypeScript + Node.js + SQLite（`node:sqlite`，无原生编译依赖），
Windows 本地直接运行，不依赖 Docker / WSL / 外部在线服务。

## 快速开始

```powershell
npm install        # 安装 devDependencies（typescript、@types/node）
npm run build      # 编译到 dist/
npm test           # 构建并运行全部测试
npm start          # 启动 HTTP API（默认自动选择空闲端口）
npm run demo       # 内置演示：low/lower/lowest 与 aaaa
```

CLI 入口为 `node dist/src/cli.js`，先 `npm run build`。

## 口径与约定

- **码点口径**：输入单词按 Unicode 码点（`for...of` / `codePointAt`）拆分为初始符号，
  不做大小写折叠、不做 Unicode 归一化（NFC/NFKC 等），输入什么就是什么。
  初始符号 ID 按码点数值升序分配，因此输入行顺序不影响训练结果。
- **ID 与文本的区别**：每个符号有唯一数字 ID。初始符号（轮次 0）的文本是单个码点；
  每次合并产生一个**新 ID**，记录左右父符号 ID、展开文本与轮次。
  符号身份是 ID（及其父符号路径），不是拼接后的字符串——
  不同合并路径即使展开文本相同，也是不同符号。
- **词内限制**：只研究词内子词构造。不做跨词合并、不做语言模型推理、
  不分词、不做词干提取、不做文本压缩。完整单词之间不插入空格或特殊结束符；
  多个词的边界由外层数组保留（编码返回数组的数组）。
- **重复词行**：相同原始字符串的权重直接累加，合并后按字符串排序，规则固定。

## 训练算法

每一轮：

1. 在当前分词状态下，枚举每个词的所有相邻符号对，按词频加权计数。
   **计数阶段保留重叠位置**：`aaaa` 有 3 个相邻对。
2. 选择加权次数最多的一对；并列时按 `(leftId, rightId)` 两个字典序最小者。
3. 将该对在每个词中**从左到右非重叠替换**为新符号：`aaaa` 中 `aa` 只替换 2 次。
   日志同时记录加权计数（`count`）与实际替换次数（`replacements`），二者不混淆。
4. 记录本轮候选排名摘要（前 10）、选中的对、加权计数、实际替换次数、词表规模。
   日志来自训练时真实语料状态，训练完成即落库，不是事后补写。

停止条件（任一满足即停，返回真实原因 `stopReason`）：

- `max_merges`：达到最大合并次数（`maxMerges=0` 合法，表示不训练任何合并）；
- `min_count`：最佳对的加权计数低于 `minCount`；
- `no_pairs`：当前已没有任何相邻对（例如全部单字符）。

## 编码与解码

- **编码**：对单个词，先映射为初始码点 ID（未知码点报 `UNKNOWN_CHAR`，
  不提供字节级后备），再**严格按学习到的合并轮次顺序**逐条应用规则，
  不重新训练、不按最长字符串贪心匹配。
- **解码**：按符号展开文本直接连接还原；未知 ID 报 `UNKNOWN_ID`。
- **轨迹**：`trace` 返回某词从初始码点拆分到每一轮合并后的完整序列。
- **版本对比**：`compare` 用两个已训练版本分别编码同一组词，输出对照。

## 持久化与版本

SQLite（默认 `./data/bpe.db`，可用 `--db` 或参数覆盖）保存：

- `models`：模型版本（同名模型每次导入语料生成新版本，旧版本保留）、
  状态（`training`/`ready`/`failed`）、算法版本（`bpe-word-v1`）、停止原因、词表规模；
- `corpus`：原始加权语料；
- `symbols`：全部符号（含父符号与轮次）；
- `merges`：每轮合并规则与候选排名摘要。

训练结果在**单个事务**内落库；训练或落库失败时回滚并将模型标记为 `failed`，
不会留下可用的半份模型。重启后仅凭数据库即可加载模型编码新词。

## 上限

| 项 | 上限 |
| --- | --- |
| 词数 | 100,000 |
| 单词长度 | 256 码点 |
| 词频 | 1,000,000,000（正整数） |
| 最大合并轮数 | 10,000 |
| 加权总量 | 不超过 `Number.MAX_SAFE_INTEGER`（精确整数） |

日志分页（`page`/`pageSize`）只影响查询，不减少实际训练轮数。

## HTTP API

默认监听 `127.0.0.1` 的**空闲端口**（启动日志打印实际端口），
也可 `--port` 指定。所有响应为 JSON；错误格式为
`{ "error": { "code": "...", "message": "..." } }`。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| GET | `/models` | 列出全部模型版本 |
| POST | `/corpus/import` | 导入语料创建新版本。Body: `{name, text:"low 5\n..."}` 或 `{name, words:[{word,weight}]}` |
| POST | `/models/:id/train` | 训练。Body: `{maxMerges?, minCount?}` |
| GET | `/models/:id` | 模型信息 |
| GET | `/models/:id/rounds?page=&pageSize=` | 分页轮次日志 |
| GET | `/models/:id/rounds/:n` | 某一轮详情（含候选排名） |
| POST | `/models/:id/encode` | Body: `{words:["low",...]}`，返回数组的数组 |
| POST | `/models/:id/decode` | Body: `{ids:[[1,2],...]}` |
| GET | `/models/:id/trace?word=lowest` | 单词合并轨迹 |
| POST | `/compare` | Body: `{a, b, words:[...]}` 两版本编码对照 |
| GET | `/models/:id/export` | 导出模型 JSON |

错误码：`INVALID_PARAM`(400)、`UNKNOWN_CHAR`(400)、`UNKNOWN_ID`(400)、
`LIMIT_EXCEEDED`(413)、`MODEL_NOT_FOUND`/`ROUND_NOT_FOUND`(404)、
`NOT_TRAINED`(409)、`INTERNAL`(500)。

## CLI

```
node dist/src/cli.js import  --name demo --text "low 5"     # 或 --file corpus.txt
node dist/src/cli.js train   --model 1 --max-merges 100 --min-count 1
node dist/src/cli.js models
node dist/src/cli.js rounds  --model 1 --page 1 --page-size 20
node dist/src/cli.js round   --model 1 --round 3
node dist/src/cli.js encode  --model 1 --words "low,lower"
node dist/src/cli.js decode  --model 1 --ids "[[11],[12]]"
node dist/src/cli.js trace   --model 1 --word lowest
node dist/src/cli.js compare --a 1 --b 2 --words "low,lower"
node dist/src/cli.js export  --model 1 --out model.json
node dist/src/cli.js serve   [--port 3000]
node dist/src/cli.js demo
```

全局参数 `--db PATH` 指定数据库文件（默认 `./data/bpe.db`）。

## 演示

`npm run demo` 依次展示：

1. `low 5 / lower 2 / lowest 3`：观察 `lo → low → lowe → lowest/lower` 公共子词逐轮出现；
2. `aaaa 1`：第 1 轮 `aa` 加权计数 3（重叠计入）但实际替换 2 次（非重叠）。

## 测试

`npm test` 覆盖：中文码点 ID 排序、重复词行累加、频数并列字典序、单字符词、
`maxMerges=0`、`minCount` 停止、未知字符/未知 ID、编码解码往返、
重排输入模型不变、旧版本不受再训练影响、训练失败无半份模型、分页不影响轮数，
以及用**独立枚举相邻位置的参考实现**逐轮核对频数与替换次数、HTTP API 端到端。

## 限制

- 仅词内 BPE，不做跨词合并与字节级后备；未知码点直接报错。
- `node:sqlite` 在 Node 25 仍带 Experimental 警告，功能可用；
  如需消除警告可换 `better-sqlite3`（需原生编译环境）。
- 训练为 O(轮数 × 语料符号数) 的朴素实现，适合教学与小中规模语料。

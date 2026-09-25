# 项目长期记忆 — 扫地机器人智能客服 Agent（D:\Agent）

## 环境与工具（每次任务都适用）

1. **Python 解释器**：本项目的依赖（langchain 全家桶、chromadb、streamlit）装在**系统 Python 3.11**：
   `C:\Users\21301\AppData\Local\Programs\Python\Python311\python.exe`
   托管默认 venv（3.13）**没有** langchain / dotenv，用它跑项目模块会 `ModuleNotFoundError`。
   → 任何 import 项目代码的验证脚本都必须用 3.11。
2. **PowerShell 里删文件不能用管道**：`Get-ChildItem ... | Remove-Item` 在 PS 5.1 下报
   `InputObjectNotBound`（对象退化成 PSObject 无法绑定参数），实测多数文件删不掉。
   可用写法：`[System.IO.File]::Delete($path)`（循环显式路径，最稳）。
3. **验证脚本通用套路**：脚本内把结果 `json.dump` 到文件 → PowerShell 运行 → 用 Read 读 JSON。
   因为 PowerShell 工具的 stdout 不会返回。
4. **外部依赖打桩**：验证 agent 逻辑时可替换模块级名字——
   `react_agent.get_chat_model`、`rag.vector_store.Chroma`、`model.factory.get_embed_model`、
   `rag.rag_service.VectorStoreService`。假 chat 模型要能参与 `|` 组合，用
   `langchain_core.runnables.RunnableLambda`。
5. **测试外部服务时的注意点**：`wttr.in` j1 的 `weather[].hourly[].time` 是 `"0"/"300"/.../"2100"`，
   小时需 `int(time)//100`。
6. **Bash 工具可用且 stdout 会返回**（2026-09-25 实测更正旧的"Bash 不可用"结论）。
   `ls`/`rm`/`grep`/`wc`/`diff`/`tr`/`md5sum`/`for` 均正常 → **需要看输出时优先用 Bash**，
   PowerShell 只在 Bash 不可用时兜底（其 stdout 不返回，需写文件再 Read）。
   同理，`.workbuddy/memory/MEMORY.md` 里"git 仓库 README.md 处于 AA 冲突"的旧结论也**已过期**：
   2026-09-25 已提交合并 `a51a2c0`。
7. **后台启动的 streamlit 实例只活当轮**（2026-09-25 实测）。用 `run_in_background` 起服务能正常
   工作一整个回合（HTTP 200），但本轮结束后进程即被回收（Duration 2m1s / status=failed）。
   → 给用户的页面链接只在当轮有效；需要人工验证页面时，**把启动命令一并给出**，别假定链接还活着。
   本机启动命令：`C:\Users\21301\AppData\Local\Programs\Python\Python311\python.exe -m streamlit run app.py --server.port 8502`

## 项目约定

- 配置分层（2026-09-25 归位，**与仓库文档/`.gitignore` 注释一致**）：
  - `config/rag.yml` = **占位模板**（`api_base: ""`、模型名用公共示例），**被 git 跟踪**，可以随便提交；
  - `config/rag.local.yml` = **真实私有值**（私有 MaaS 端点 + `qwen3.8-max` / `qwen3.7-text-embedding`），
    **已被 `.gitignore` 忽略，绝不要提交**；
  - `utils/config_handler.load_rag_config()` 先读 rag.yml 作基线，再用 rag.local.yml **浅覆盖**同名键；
  - `config/rag.yml.example` 与 `rag.yml` 内容相同，作为模板备份保留。
  ⚠️ 端点/自定义模型名只写在 `rag.local.yml`；工作记忆里的相关记录也已脱敏为 `ws-<workspace-id>`，
  不要把明文写回任何被跟踪的文件。密钥走环境变量或根目录 `.env`。
- **报告模式的唯一权威源是会话状态** `session["mode"]`（`ReactAgent.sessions`，按
  `conversation_id` 隔离），持久化为 `conversations.mode`；`get_mode()` / `_get_system_prompt()`
  一律按 `conversation_id` 读取。中间件 `_context_store` 只是按 user_id 的**进程内存镜像**，
  仅用于其对外 HTTP 接口展示，**不参与提示词切换决策**（2026-09-25 修正，此前是双源且消费方读错）。
- 报告模式在最终回答产出后自动退出；达到最大迭代次数的收尾分支同样会复位（2026-09-25 补）。
- 会话库 `data/sessions.db` 开 WAL；`SessionStore` 所有连接经 `_connection()` 上下文管理器收口。
- 知识库入库已从"首次提问时懒加载"改为 **app 启动时预热**（`app.py::warm_up_knowledge_base`，
  `@st.cache_resource`）。改动原因见下方 Chroma 运维铁律；仍可按 MD5 增量去重，幂等。
- 启动：`streamlit run app.py`；可选 `python middleware.py` 起中间件 HTTP 服务（127.0.0.1:8000）。
- 本机系统环境变量已设置 `OPENAI_API_KEY` 与 `DASHSCOPE_API_KEY`（均为 DashScope 风格 key），
  因此端点必须由 `rag.yml` 的 `api_base` 提供，否则会回落到 OpenAI 官方地址导致鉴权失败。

## 会话历史完整性（P2，2026-09-25 修复）

**症状**：某条会话打不开/怎么问都失败，日志里是
`OpenAIInvalidRequestError: 400 - invalid_parameter_error: The provided messages input is invalid.
The error info is [Can only get item pairs from a mapping.]`，**每次重试都复现**。

**机理**：ReAct 循环里 `AIMessage(tool_calls=[...])` 与其配对的 `ToolMessage` 若分两次落盘，
进程在两次之间中断（崩溃 / Streamlit Stop / 强杀 / 容器被回收）就会留下"声明了工具调用、
却没有配对 tool 响应"的历史。这种历史原样发给模型 → 400。**危险窗口 = 第一个工具的执行时长**
（RAG 类工具动辄几十秒，窗口很大）。

**两层修复**（`agent/tools/session_store.py` + `agent/react_agent.py`）：
1. **写入原子化**：`chat()` / `chat_stream()` 改为「先把本轮全部工具执行完并收集配对响应，
   最后把 AI 消息 + 它的全部 tool 响应一次性 append 并落盘」。工具执行期间历史末尾仍是用户消息，
   所以任何落盘点（含 `_switch_to_report_mode` 的落盘）都是自洽的。
2. **恢复时清洗**：`session_store.sanitize_history()`（幂等、不修改入参），在 `get_conversation()`
   里对恢复出来的历史统一调用。规则：无配对响应的 tool_call 从声明中剔除；剔除后既无 tool_call
   又无正文的 AI 消息整条丢弃；找不到前置声明的孤儿 tool 消息丢弃。
   已经损坏的旧记录**读取时即自愈**，不需要一次性数据迁移。

**验证脚本**：根目录 `_verify_p2.py`（A 单元 8 例 / B 真实库 25 条自愈 / C 真模型 A/B /
D+D2 `chat` 与 `chat_stream` 双路径崩溃模拟），报告 `_p2_report.json`。
⚠️ 项目自带的 `test_react_integration.py` / `test_session_persist.py` **完全不覆盖 `chat_stream`**，
改流式路径必须自己补验证。

## 会话状态机孤儿（P3，2026-09-25 修复）

**症状**：某用户切到别的用户、或中途关掉浏览器后，之前那条会话**从历史列表里消失**，
既看不到也清不掉，对话内容永久不可见。

**机理**：`active` 与 `archived` 构成的状态机里，"当前会话"同时被 **属主(user_id)** 和
**状态(status)** 两个条件筛选 —— `list_archived(user_id)` 只查 `status='archived'`，
而归档动作 `archive_all_active(user_id)` 也只作用于当前用户。两者口径一致地"漏掉"
了**属主已切走的 active**：它不是 archived（列表不显示），也不是当前用户的 active（不会被归档）。

**修法**：让"同一时刻至多一条 active"成为不变量。
- `session_store.archive_all_active(user_id=None)`：`None` 时归档**全部用户**的 active；
  传 `user_id` 时保持原语义（`test_react_integration.py` 仍在用）。
- `app.py::start_new_conversation` 改为无参调用（页面每次打开/切用户都会走一次）。
- 已被遗弃的 active 会在**下次页面打开时自动收起**并出现在对应用户的历史列表里，无需迁移脚本。

**代价（已知并接受）**：多浏览器场景下，B 标签页打开页面会把 A 标签页正在进行的会话一并归档。
不丢数据（消息仍按 conversation_id 追加），只是可能提前收起。单用户演示定位下可接受。

**验证脚本**：根目录 `_verify_p3.py`（A 语义 4 例 / B 旧行为 vs 新行为对照 / C app 调用点静态检查 /
D 真实库只读体检），报告 `_p3_report.json`。

## 会话库清理与备份（2026-09-25）

- 已按用户确认清理：删掉 user 1001 的 117 条 archived 测试残留（id 41~163，含 25 条悬空记录），
  保留当前 active 空会话。**保留 active 是必须的**：删掉它会让运行中的页面在
  `save_conversation` 里 `SELECT ... WHERE id=?` 拿不到行 → 走 `return` 分支**静默丢弃消息**。
- 清理前全量快照：`.workbuddy/backups/sessions.db.20260925_170729.pre-cleanup.bak`
  （118 会话 / 559 消息 / 117 archived）。该目录已被 `.gitignore` 忽略。
- **备份要用 sqlite 的 `VACUUM INTO` 而不是 `cp`**：WAL 模式下已提交但未 checkpoint 的页
  可能仍在 `-wal` 里，只拷主文件会丢数据。用 `mode=ro` 打开备份文件还会顺手生成
  `-shm`/`-wal`，记得清掉。

## Chroma 向量库运维铁律（2026-09-25 故障后沉淀）

1. **写入与读取不能并发**。Chroma 是单进程嵌入式向量库，一边 `add_documents`（写）
   一边 `retriever.invoke()`（读）会抛
   `chromadb.errors.InternalError: Error creating hnsw segment reader: Nothing found on disk`。
   → 入库只在启动阶段做（单线程），用户请求路径只读。
2. **看到该报错先别删库**。磁盘索引大概率是好的，只是**客户端句柄失效**。
   判据：`collection.count()` 与 k 大值检索能否召回**最后入库的文件**。
   修复：重建 `VectorStoreService()` 换句柄即可（`rag_service.retriever_docs` 已内置自愈重试）。
3. **hnsw 文件大小 ≠ 向量条数**。`header.bin` 里的 `100` 是预分配容量；
   实测 `count()=335` 而 `data_level0.bin` 仅 423600B。**永远用 `count()` 判断**。
4. **embedding 单次上限 20 条**（阿里云 text-embedding），批量写入取 `batch_size=10`。
5. 诊断顺序：`md5.txt` 行数 ↔ `embedding_metadata` 按 source 分组计数 ↔ `max_seq_id`
   ↔ `collection.count()` ↔ k=40 检索召回分布。以上全健康 ⇒ 问题在运行时，不在数据。

## 工具健壮性 / 常见反模式（2026-09-25）

### `dict.get(key, default)[0]` 是陷阱（get_weather 已踩）
`current.get("lang_zh", [{}])[0]` —— **`get` 的默认值只在 key 缺失时生效**，key 存在但值为
**空列表**时 `[0]` 照样 `IndexError`。wttr.in 的 `current_condition[0].lang_zh` 确实会返回 `[]`。
正确写法：`(current.get("lang_zh") or [{}])[0]`。
- 后果放大：`get_weather` 外层是 `except Exception` → 整个工具降级成"天气信息暂时无法获取"，
  而同一份响应里 `weatherDesc` 明明有可用英文描述。**静默降级比异常更难发现。**
- 已一并加固 `weatherDesc` 兜底 + 两者都取不到时输出"未知"，不丢弃温度/湿度等有效信息。
- ⚠️ **项目自带的 `verify_tools.py::weather_check` 原断言只查"结果含城市名"，
  而降级文案"城市深圳天气信息暂时无法获取"同样含城市名 → 这个 bug 能静默通过校验。**
  已补强断言（禁"暂时无法获取"/"天气为None"/"天气为未知"，要求含"摄氏度"+"湿度"）。
  → 教训：校验脚本的断言必须能区分"成功"与"降级"，否则等于没测。
- 同类写法全局排查：`middleware.py:156` `params.get("user_id", [None])[0]` 是**同款但不会触发**
  （`parse_qs` 对存在的 key 永远至少给一个值，实测 `?user_id=` 直接被丢弃 → key 缺失 → 走默认值），
  故未改动。

## 尚未完成的技术债（2026-09-25 审计结论，详见根目录《代码审计与优化建议.md》）

### ~~结论：报告模式状态是双源的，且消费方读错了那一套~~（2026-09-25 已修复）
原问题：`conversations.mode`（DB，按 conversation_id）与 `middleware._context_store[user_id]`（进程内存）
各写各的，而 `_get_system_prompt(user_id)` 读的是中间件 → 进程重启后报告会话用错提示词，
`finish_report` 还会被误触发。
**已改**：`session["mode"]` 成为唯一权威源，`_get_mode` 改名公开的 `get_mode(conversation_id)`，
`_get_system_prompt` 改收 `conversation_id`，中间件降级为镜像；`app.py` 侧边栏模式指示器与
退出按钮也改读/改走会话状态。回归验证脚本：根目录 `_verify_mode.py`（5 场景，桩模型不调 API）。

### ~~报告功能在当前时间下拿不到数据~~（2026-09-25 已修复）
`data/external/records.csv` 只覆盖 2025-01~2025-12（用户 1001~1010），而 `get_current_month()`
返回真实系统时间（2026-09）。**已改**：`fetch_external_data` 回退到语义最近的有数据月份，
返回 JSON 带 `实际月份`/`用户请求月份`/`数据说明`；`report_prompt.txt` 要求报告开头声明实际数据月份。
验证脚本：根目录 `_verify_p01.py`。

### 其他 P0
`report_prompt.txt` 缺 step 约束（**2026-09-25 已补**）：详见下方"报告提示词"一节，
注意原判据"日志已复现第 2 轮收尾"**证据不成立**；
`_execute_tool` 需防御空工具名（流式 chunk 曾产生 `name=''`），`rag_summarize` 需短路空 query；
`get_weather` 的 `lang_zh` 空列表 IndexError **2026-09-25 已修**（见下方"工具健壮性"）。

### P1/P2（2026-09-25 第二轮逐条核查后更新）
**`save_conversation` 全量重写 O(n²)**（**2026-09-25 已修**）：原缺陷 —— 每次调用先
`DELETE` 全部消息再全量 `INSERT`，一轮对话触发 4~6 次，单次成本 O(历史长度) ⇒ O(n²)。
修复：新增 `SessionStore.append_messages(conv_id, mode, new_messages)`，seq 从既有
`MAX(seq)+1` 接续、只写新增段；`ReactAgent._persist_session` 按 `session["_saved_len"]`
切片调用（`_init_session` 恢复时把该值初始化为恢复出的历史长度）。`save_conversation`
保留给"覆盖既有历史"的场景（测试预置 / 外部调用方），语义未变。
实测：301 条历史的会话跑一轮对话，SQL 影响行数 **2432 → 8**（304×）；验证脚本
根目录 `_verify_p4.py`（A 等价性 / B 写放大 / C 边界 / D 真 Agent 桩模型 / E 与 sanitize 共存）。
⚠️ **`_saved_len` 前提是"历史只追加不截断/重排"** —— 已 grep 全项目确认
（`react_agent.py` 只有 `append`/`extend`）；若将来出现截断逻辑，`_persist_session`
的 `saved > len(history)` 防御分支会回退全量写，但正常截断需要显式同步该计数。

**`ReactAgent.sessions` 仍经 `st.cache_resource` 进程级共享且无淘汰**（未改）：会话数由
conversation_id 决定，单用户演示规模下无实际影响，属已知设计取舍 —— 要改就得引入 LRU 并
处理"被淘汰会话再次提问"的恢复路径，收益不足；
**中间件「HTTP 服务」**（2026-09-25 已加固）：`HTTPServer` → `ThreadingHTTPServer`、
`fill_context` 的 `ctx.update(extra)` 加**受控字段白名单**（请求体不能再覆盖
`mode`/`prompt_type`/`user_id`，非受控自定义字段照常合并）、`_read_json` 遇到非法 JSON
按空处理返回 400 而非冒泡 500。**仍无鉴权**（有意保留：默认只绑 127.0.0.1，且中间件已非
模式权威源，加 token 会牵动 agent 侧调用契约，收益不足）；
**悬空 tool_calls**（**2026-09-25 已修**）：原缺陷 —— 先落盘 `AIMessage(tool_calls)`、之后才逐条
append `ToolMessage`，恢复时不清洗 → 中断即留下缺配对 tool 响应的历史，恢复后调 API 直接 400 且持续复现。
修复见下方「会话历史完整性（P2）」一节；DB 里 25 条实例（id 75~99）已在读取时自动清洗；
**跨用户孤儿会话**（**2026-09-25 已修**）：原缺陷 —— `start_new_conversation(user_id)` 只归档
当前用户，`list_archived` 只查 `status='archived'` → 旧用户 active 会话不可见也不清理。
修复见上方「会话状态机孤儿（P3）」一节；
**缓存无上限**（**2026-09-25 已修**）：原缺陷 —— `_rag_cache`/`_weather_cache` 只有 TTL 判断，
过期条目永不删除、字典无容量上限 ⇒ 常驻进程内存无界增长。修复：新增
`agents_tools._cache_put(cache, key, value, ttl)` + `_CACHE_MAX_ENTRIES=128`，
写入时先清过期项、仍超上限则按写入时间淘汰最旧，两个缓存统一走它；
**日志无轮转**（**2026-09-25 已修**）：`logger_handler.py` 由 `FileHandler` 改为
`RotatingFileHandler(maxBytes=5MB, backupCount=5)`（注意须 `import logging.handlers`，
只 `import logging` 不带子模块）；
**`_next_6h_rain_desc` 的时段判据**（**2026-09-25 已修**）：原 `start >= now_hour` 有**两个**
方向的错 —— 既漏掉进行中的 3 小时时段，又**取到窗口之外的数据**（实测 now=04:30 报 40%
来自 09:00 段）。改为 `start + 3 > now_hour`；同时 `chanceofrain` 用 `or 0` 兜空串
（空串会让 `int("")` 抛异常 → 整段降雨描述被静默吞掉）。**取 2 个时段而非 3 个是有意的**：
3 个会把窗口拉到约 9 小时，`max()` 出的概率被系统性高估；
**工具边界**（**2026-09-25 已修**）：`_execute_tool` 的 `name` 取 `or ""`、`args` 取 `or {}`
（`get("args", {})` 在值为 `None` 时仍返回 `None`，直接 `invoke(None)` 会抛错），空工具名
单独给"请重新选择工具"而非误导性的"工具 '' 不存在"；`rag_summarize` 对空/纯空白 query
短路，不进 embedding 也不进模型；
**requirements.txt**（**2026-09-25 已修**）：补 `dashscope==1.27.3`（`model/factory.py:36` 用的
`langchain_community.embeddings.DashScopeEmbeddings` 运行时必需），并把全部 `>=` 改为
**实测版本 `==`**（Python 3.11 跑通的组合记录在文件头注释）；
**`eval_questions.py` 源码丢失（只剩 .pyc）** —— 仍未处理。

### 已核实为误报 / 已修（2026-09-25 二轮）
- 首次 RAG 阻塞 28 秒：**已修**，预热挪到 app 启动。但代价是"挪走"不是"消除"——冷启动页面
  仍要等。**2026-09-25 二次实测（热盘）**：`import rag.vector_store`（=chromadb 等）**8.70s**
  + `import rag.rag_service` 0.10s + `RagSummarizeService()`（Chroma 客户端 + embedding 构造
  + 6 文件 MD5 检查）**5.67s** = **14.47s**（早前 38.56s 那次是冷盘/首次）。
  → **决定不改后台线程预热**：① 这 15~40 秒每进程只付一次（`st.cache_resource`），非每次交互；
  ② 后台预热会重新引入"写 Chroma 与读 Chroma 并发"，正是 `hnsw segment reader: Nothing found
  on disk` 那个已修重大故障的成因，**收益与风险不成比例**；③ 大头是第三方库导入成本，无低成本解法。
  已改为**让等待可预期**：`app.py` spinner 明说"仅首次启动需等待约 15~40 秒"。
  若日后要压这部分，正确做法是单独立项：RAG 拆成独立进程（Chroma 单写者）+ agent 侧走 HTTP，
  顺带根治"读写不能并发"的约束 —— **不要混在杂项里做**。
- `app.py:241` 注释"输入提交后会自动 rerun"：**已修**，现为 `app.py:296-299`，并补了 `st.rerun()`。
- `config/rag.local.yml` 的 `qwen3.8-max`：**前提不成立**，该文件当时不存在，模型名写在 `config/rag.yml`，
  端点为私有 MaaS，模型名可自定义，能正常初始化。
  → **2026-09-25 已按设计意图归位**：`rag.yml` 还原为占位模板（`api_base: ""`）+ `rag.local.yml` 存真实值
  （已被 .gitignore 忽略）+ 恢复 `config/rag.yml.example`。详见「配置分层」一节。
- git 冲突：2026-09-25 已解决并提交合并 `a51a2c0`（README.md 工作区版早已无冲突标记，
  只是没 `git add`）。⚠️ 当时遗留的 13 个已跟踪文件改动**已于同日固化为基线提交**：
  `7c78b58`（工作记忆同步）+ `897628b`（代码修复基线），工作区已跟踪文件干净。

### 无价值但会误导人的死代码（2026-09-25 已清）
`app.py` 的 `pending_switch_user` 从未被写入过。**已删除**，只保留 `user_id_input`
默认值预置（并改写注释说明"为什么不用 text_input 的 value= 参数"）。
同时修掉同一类的"渲染滞后一轮"：`st.button` 点击本身就是那次 rerun，但处理器运行时
**它上方的组件已经渲染完**，所以侧边栏模式指示器会显示上一会话的模式。已在
「🆕 新对话」与「历史会话载入」两处处理器末尾补 `st.rerun()`，并把
"点击后 Streamlit 会自动 rerun，无需手动调用"这条**看似正确其实误导**的注释改写清楚。
（对话路径的同类问题更早已修，见上条。）

## 报告提示词 report_prompt.txt（2026-09-25 重写）

- 原版问题：**报告模式一旦激活，强约束就消失**——四步取数流程只写在 `main_prompt.txt`，
  而 `fill_context_for_report` 一触发就切到 `report_prompt.txt`，此刻正是最需要约束（数据还没取）的时候。
  原首句"根据查询回来的信息，写一份…"也容易被读成"信息已就绪"。
- 已补：首句明确"进入报告模式≠数据已就绪"+「报告数据获取强约束」四步（get_user_id → get_current_month
  → `fetch_external_data` 拿不到结果不得写正文 → 可选 rag_summarize），并声明
  `fill_context_for_report` 已触发、**不要重复调用**；输出要求加"数据必须来自工具返回，不得估算编造"。
- ⚠️ **旧判据说"日志已复现第 2 轮收尾、fetch_external_data 从未被调用"，经查证证据不成立**：
  「回答长度 9」那批日志全挤在同一秒内（`agent_20260923.log` 18:46:08 / 20:22:58），
  且出现"用户 1003 没有任何用户消息就被切到报告模式"这种只有打桩/批量脚本才有的形态。
  真实模型记录中 5 个报告轮次**全部**先调了 `fetch_external_data`
  （09-11 两轮、09-25 的 10:46 / 10:57 / 11:09 三轮），未出现过跳过取数直接写报告。
- A/B 实测（真实模型，危险入口=新建会话后先置 report 模式再提问「生成我的使用报告」）：
  新旧提示词**都**正确走出 get_user_id+get_current_month → fetch_external_data → rag_summarize，
  都声明了"基于 2025-12 的数据"。新版回答开头干净（旧版有 `I'll gather the necessary information first.`
  这类多余英文行），约束从"靠模型自己推断"变成"提示词显式要求"。
- 结论：这是一处**结构性缺陷的防御性加固**，不是"已复现 bug 的修复"。

## 网页端状态检验（2026-09-25 补齐）

- 侧边栏有折叠面板「🔍 状态自检（调试）」，并排显示 `权威模式（conversations.mode）` 与
  `中间件镜像（user_id 进程内存）`，不一致时给 warning —— 这是页面上唯一能直接看出"双源"的地方。
- **报告轮结束时模式一定已自动复位 normal**，所以「📊 当前：报告模式」只在轮次被中断时出现
  （模型报错 / Streamlit Stop / 浏览器刷新）。正常跑完的报告轮侧边栏显示 💬 普通模式，不是 bug。
- 无头渲染验证用 Streamlit 自带 `streamlit.testing.v1.AppTest.from_file("app.py")`，
  可直接断言 `at.exception` / `at.sidebar.*`，比手动点浏览器可靠。
  ⚠️ 但它会跑 `start_new_conversation` → 全局 `archive_all_active()`，**会动真实会话库**。
  **2026-09-25 起已解决**：`SessionStore.__init__` 支持 `AGENT_SESSION_DB` 环境变量覆盖
  （仅供测试，生产不设该变量即走 config 默认路径）；`test_app_history.py` 已在文件顶部
  指向临时目录并在 finally 里删除。验证方式：跑测试前后比对 `data/sessions.db` 的**整文件 MD5**。
  → 现在可以安全地跑页面级测试了（此前记录"不要跑 AppTest"的约束**已过期**）。
- AppTest 用法坑：`at.session_state` 是 `SafeSessionState`，**没有 `.get()`**（要用
  `try/except KeyError`）；旧 run 树的按钮对象点击会失效，每次点击前须重新收集。
  完整清单见技能 `streamlit-apptest-safe-verification`。
- 复现"权威 report vs 镜像 normal"分歧的原样本 `conversations.id=153` **已于 2026-09-25 清理**，
  需要时在页面上重新问一次「生成我的使用报告」并中途打断即可再造。
- ⚠️ **不要擅自杀用户的 streamlit 进程**（2026-09-25 17:02 用户自己起了 8502）。
  需要页面验证时把启动命令交给用户，让他们自己重启（改提示词/改模块级函数后必须重启进程，
  `st.cache_resource` 缓存着旧实例，刷新浏览器无效）。

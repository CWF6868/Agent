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

## 项目约定

- 配置分层：`config/rag.yml` 是**唯一实际生效**的配置文件（内含真实 `api_base`）。
  代码支持 `config/rag.local.yml` 覆盖同名键，但**该文件当前并不存在**（2026-09-25 核实，
  `config/` 下只有 agent/chroma/prompts/rag 四个 yml）。密钥走环境变量或根目录 `.env`。
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
`save_conversation` 全量重写 O(n²)（`session_store.py:317-326`，实测成立）；
`ReactAgent.sessions` 经 `st.cache_resource` 进程级共享且无淘汰；
中间件「HTTP 服务」模式形同虚设且 `HTTPServer` 非线程化、无鉴权、
`fill_context` 的 `ctx.update(extra)` 可被请求体覆盖 `mode`；
**悬空 tool_calls**：`react_agent.py:360-361` 先落盘 `AIMessage(tool_calls)`，之后 384-389 才逐条
append `ToolMessage`；`_init_session` 恢复时不清洗 → 中途中断即留下缺配对 tool 响应的历史，
恢复后调 API 直接 400 且持续复现。**代码缺陷成立**（DB 里 25 条实例是批量脚本产物，非真实崩溃）；
**跨用户孤儿会话**：`start_new_conversation(user_id)` 只归档当前用户，`list_archived` 只查
`status='archived'` → 旧用户 active 会话不可见也不清理（代码路径成立，暂无实例）；
缓存无上限（`_rag_cache`/`_weather_cache` 只有 TTL、无淘汰，过期条目永不删除）；
日志无轮转（`logger_handler.py:45` 是 `FileHandler` 非 `RotatingFileHandler`，实测单日 ~100KB）；
`_next_6h_rain_desc` 的 `>= now_hour` 会漏掉进行中的 3 小时时段、**并把它之后的窗口数据当"未来6小时"上报**
（实测 now=04:30 报 40% 来自 09:00 段）；requirements.txt 缺 `dashscope`（`model/factory.py:36` 用的
`langchain_community.embeddings.DashScopeEmbeddings` 运行时必需）且全部 `>=` 未锁版本；
`eval_questions.py` 源码丢失（只剩 .pyc）。

### 已核实为误报 / 已修（2026-09-25 二轮）
- 首次 RAG 阻塞 28 秒：**已修**，预热挪到 app 启动。但代价是"挪走"不是"消除"——实测首次
  `_get_rag()` 仍需 **38.56s**，拆解后主体是 `import rag.vector_store`（8.41s）等**模块导入 +
  客户端初始化**，不是入库（日志显示 6 个知识文件全部 MD5 命中跳过）。冷启动页面仍要等。
- `app.py:241` 注释"输入提交后会自动 rerun"：**已修**，现为 `app.py:296-299`，并补了 `st.rerun()`。
- `config/rag.local.yml` 的 `qwen3.8-max`：**前提不成立**，该文件不存在，模型名在 `config/rag.yml`，
  端点为私有 MaaS，模型名可自定义，能正常初始化。
- git 冲突：2026-09-25 已解决并提交合并 `a51a2c0`（README.md 工作区版早已无冲突标记，
  只是没 `git add`）。⚠️ 另有 **13 个已跟踪文件的修改仍未提交**，待在后续修复中一并处理。

### 无价值但会误导人的死代码
`app.py:40-47` 的 `pending_switch_user` 从未被写入。
（"输入提交后会自动 rerun"那条错误注释已于 2026-09-25 修正：在对话收尾补了 `st.rerun()`，
否则侧边栏模式指示器会滞后一轮。）

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
- 现成参考样本：`conversations.id=153`（标题「生成我的使用报告」，`mode='report'`，archived）。
  打开它即可复现"权威 report vs 镜像 normal"的分歧场景。
- 无头渲染验证用 Streamlit 自带 `streamlit.testing.v1.AppTest.from_file("app.py")`，
  可直接断言 `at.exception` / `at.sidebar.*`，比手动点浏览器可靠。
- ⚠️ `data/sessions.db` 里 user 1001 有 116 条 archived 会话，多为历次测试脚本残留，
  侧边栏历史列表很长；未擅自清理，待用户确认。

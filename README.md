# 🤖 扫地机器人智能客服 Agent

基于 **ReAct 架构** 的扫地/扫拖机器人智能客服系统。融合 **RAG 知识库检索**、**多工具自主调用**、**报告模式提示词切换** 与 **多会话持久化**，提供「Streamlit 可视化 Web 界面 + 可独立调用的 Agent 内核」双入口。

用户可以咨询使用保养、故障排查、天气适配等问题，也可以一句话生成个性化的「个人使用报告」。

---

## ✨ 功能特性

| 特性 | 说明 |
| --- | --- |
| 🧠 **ReAct 思考循环** | 思考 → 调用工具 → 观察结果 → 再思考 → 最终回答，最多迭代 5 轮 |
| 📚 **RAG 知识增强** | Chroma 向量库 + DashScope Embedding，检索扫地机器人专业资料后再回答 |
| 🛠️ **7 个可调用工具** | 知识检索、天气查询、用户身份、月份获取、外部数据、报告上下文注入等 |
| 📊 **双提示词模式** | 普通客服模式 / 报告写手模式，按 `conversation_id` 由会话状态切换（中间件仅作镜像，供其对外接口展示） |
| 💬 **多会话持久化** | SQLite 存储，支持新建 / 归档 / 恢复 / 删除历史会话 |
| ⚡ **流式输出** | 模型回答逐字渲染，工具调用过程可折叠查看 |
| 🚀 **启动预热 + 惰性兜底** | 知识库在启动阶段预热（仅首次启动需等待约 15~40 秒），模型等其余组件首次使用时才初始化 |
| 🔌 **中间件双模式** | 支持「本地函数模式」（同进程）与「HTTP 服务模式」（独立部署） |

---

## 🏗️ 系统架构

```mermaid
flowchart TB
    subgraph UI["前端层 (Streamlit)"]
        A[app.py<br/>对话界面 / 侧边栏 / 历史会话]
    end

    subgraph Core["Agent 核心层"]
        B[ReactAgent<br/>ReAct 循环 + 提示词切换]
    end

    subgraph Tools["工具层"]
        C1[rag_summarize]
        C2[get_weather]
        C3[get_current_month]
        C4[fetch_external_data]
        C5[get_user_id]
        C6[get_user_location]
        C7[fill_context_for_report]
    end

    subgraph Support["支撑层"]
        D1[模型工厂<br/>ChatOpenAI / Embeddings]
        D2[向量库服务<br/>Chroma + 文本切分]
        D3[会话存储<br/>SQLite]
        D4[中间件<br/>报告上下文管理]
    end

    A -->|chat_stream 事件流| B
    B --> C1 & C2 & C3 & C4 & C5 & C6 & C7
    C1 --> D2
    C7 --> D4
    B --> D1
    B --> D3
    A --> D3
    D2 --> D1

    style A fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    style B fill:#fff3e0,stroke:#f57c00,color:#e65100
    style D4 fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
```

### 一次「生成使用报告」的完整链路

```mermaid
sequenceDiagram
    participant U as 用户
    participant A as app.py
    participant R as ReactAgent
    participant M as 中间件
    participant T as 工具集

    U->>A: "生成我的使用报告"
    A->>R: chat_stream(...)
    R->>T: get_user_id() → "1001"
    R->>T: get_current_month() → "2025-06"
    R->>T: fill_context_for_report()
    T->>M: 注入报告上下文（mode=report）
    M-->>R: 模式切换为 report
    Note over R: 后续系统提示词切换为<br/>报告写手提示词
    R->>T: fetch_external_data("1001","2025-06")
    T->>T: rag_summarize("保养建议")
    R-->>A: 流式输出 Markdown 报告
    A-->>U: 实时渲染报告
```

---

## 📁 项目结构

```
Agent/
├── app.py                      # Streamlit 主入口（对话界面 + 侧边栏 + 历史会话）
├── verify_tools.py             # 工具运行状态检查脚本（4 层检查）
├── test_session_persist.py     # 会话持久化验证（不依赖模型/RAG）
├── test_react_integration.py   # ReactAgent 集成验证（假模型）
├── test_app_history.py         # Streamlit AppTest 交互流程验证
├── md5.txt                     # 知识库同步标记（运行时自动生成、按实际结果重写，非权威数据源）
│
├── agent/                      # Agent 核心
│   ├── react_agent.py          # ReAct 主体：思考循环 / 流式输出 / 模式切换
│   └── tools/
│       ├── agents_tools.py     # 工具集定义（无状态 + 有状态闭包工厂）
│       ├── session_store.py    # SQLite 多会话存储 + 旧表自动迁移
│       └── middleware.py       # 报告上下文中间件（本地函数 + HTTP 服务）
│
├── model/
│   └── factory.py              # 模型工厂：ChatOpenAI / DashScopeEmbeddings 惰性单例
│
├── rag/                        # 检索增强
│   ├── rag_service.py          # RAG 总结服务（检索 + 拼上下文 + 模型总结）
│   └── vector_store.py         # 向量库服务（Chroma + 切分 + 按文件 MD5 增量同步）
│
├── utils/                      # 通用工具
│   ├── config_handler.py       # 统一加载 4 个 yml 配置
│   ├── path_tool.py            # 项目绝对路径解析
│   ├── file_handler.py         # MD5 计算 / 文件列举 / PDF·TXT 加载器
│   ├── logger_handler.py       # 日志器（控制台 + 文件按日切分，单文件 5MB 轮转、保留 5 份）
│   └── prompt_loader.py        # 提示词加载
│
├── prompts/                    # 提示词模板
│   ├── main_prompt.txt         # 普通客服系统提示词
│   ├── report_prompt.txt       # 报告写手系统提示词
│   └── rag_summarize.txt       # RAG 总结提示词（含 {input} / {context}）
│
├── config/                     # 配置
│   ├── rag.yml                 # 模型名 / embedding 名 / API 端点
│   ├── rag.yml.example         # 配置模板（脱敏，供 clone 后复制）
│   ├── chroma.yml              # 向量库 / 切分 / 数据目录配置
│   ├── agent.yml               # 外部数据路径 / 会话数据库路径
│   └── prompts.yml             # 提示词文件路径
│
├── requirements.txt            # Python 依赖清单
├── .env.example                # 环境变量模板
├── .gitignore                  # Git 忽略规则
├── LICENSE                     # MIT 许可证
│
├── data/                       # 知识库与数据
│   ├── *.txt / *.pdf           # 扫地机器人知识文档
│   ├── external/records.csv    # 用户使用记录（10 用户 × 12 月）
│   └── sessions.db             # SQLite 会话库（运行时生成）
│
├── chroma_db/                  # Chroma 持久化目录（运行时生成）
└── logs/                       # 运行日志（运行时生成）
```

---

## 🚀 快速开始

### 1. 环境要求

- Python **3.10+**（推荐 3.11）
- 可访问大模型 API 的网络环境
- 可选：DashScope（阿里云百炼）API Key —— 用于 Embedding

### 2. 安装依赖

```bash
pip install streamlit langchain langchain-core langchain-openai langchain-community \
            langchain-chroma langchain-text-splitters chromadb pyyaml pypdf
```

### 3. 配置

复制配置模板并填入你的真实信息：

```bash
cp config/rag.yml.example config/rag.local.yml
```

编辑 `config/rag.local.yml`：

```yaml
chat_model_name: qwen-max            # 对话模型名（按所用服务商填写）
embedding_model_name: text-embedding-v3
api_base: "https://your-endpoint/v1" # 留空则走 OpenAI 官方默认
```

> 也可以直接编辑 `config/rag.yml`（仓库内的默认值即为无端点配置）。

设置密钥环境变量：

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-xxxxxxxx"

# Linux / macOS
export OPENAI_API_KEY="sk-xxxxxxxx"

# Embedding 使用 DashScope 时需要
export DASHSCOPE_API_KEY="sk-xxxxxxxx"
```

> **API 端点优先级**：`config/rag.yml` 的 `api_base` > 环境变量 `OPENAI_API_BASE` > `OPENAI_BASE_URL` > OpenAI 官方默认。

### 4. 构建知识库（首次运行）

将知识文档（`.txt` / `.pdf`）放入 `data/` 目录，然后执行：

```bash
python rag/vector_store.py
```

程序会遍历 `data/` 下所有允许类型的文件（后缀大小写不敏感），按文件 MD5 判断该做什么：

| 情况 | 行为 |
| --- | --- |
| 新增文件 | 切分入库 |
| 文件内容变了 | **先删除该文件的旧分片，再入库新分片**，不会留下过期内容 |
| 文件从 `data/` 删除 | 其分片与 MD5 记录一并清除 |
| 文件还在，但内容已无法入库<br>（被清空 / 只剩空白 / 编码损坏） | 同样**清除其旧分片**，不让上一版内容继续被检索到；该文件不写入 MD5 记录，修好后下次启动会自动重新入库（修好前每次启动留一条告警，避免"改了文件却没生效"被静默吞掉） |

一句话概括同步的不变量：**向量库里任何一个文件的分片，必须对应磁盘上该文件的当前内容。**

重复执行是幂等的：内容没变的文件直接跳过，不会重复写入。

> 调整切分参数请编辑 `config/chroma.yml` 的 `chunk_size` / `chunk_overlap` / `separators`。
> 若需完全重建，建议**同时**删除 `md5.txt` 与 `chroma_db/` 后重跑。
> 只删其中一个也不会静默失效：`md5.txt` 只是"该文件已同步"的加速标记、不是权威数据源，
> 程序每次同步都会校验它与向量库是否一致、并按本次实际结果重写它
> （只删 `chroma_db/` → 检测到库内无分片 → 重新入库；只删 `md5.txt` → 记录为空 → 全量重新入库）。

### 5. 启动应用

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`。

---

## 💡 使用方式

### 界面操作

| 区域 | 功能 |
| --- | --- |
| **用户 ID** | 隔离不同用户的历史会话与使用记录（默认 `1001`） |
| **所在城市** | 用于天气查询与地域适配 |
| **中间件设置** | 切换「本地函数」/「HTTP 服务」模式 |
| **模式指示器** | 实时显示当前是 💬 普通模式 还是 📊 报告模式 |
| **🆕 新对话** | 归档当前会话并开始新会话（历史记录保留） |
| **📚 历史会话** | 点击可恢复全部对话并继续；🗑 可删除 |

### 示例提问

```
扫地机器人每天用合适吗？
深圳最近天气潮湿，对扫地机器人有影响吗？
扫地机器人边刷不转了怎么排查？
生成我的使用报告
生成我 2025 年 6 月的使用报告
```

### HTTP 中间件模式（可选）

若需将中间件独立部署：

```bash
python agent/tools/middleware.py
# 服务启动于 http://127.0.0.1:8000
```

然后在 Streamlit 侧边栏选择「HTTP 服务」并填写地址。

**中间件接口：**

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/context/fill` | 注入报告上下文，body: `{"user_id": "1001"}` |
| `GET` | `/context/get?user_id=1001` | 查询用户当前上下文 |
| `POST` | `/context/clear` | 清理上下文，恢复普通模式 |
| `GET` | `/health` | 健康检查 |

---

## 🧩 工具清单

| 工具名 | 入参 | 说明 |
| --- | --- | --- |
| `rag_summarize` | `query: str` | 从向量库检索扫地机器人专业资料并总结（结果缓存 10 分钟） |
| `get_weather` | `city: str` | 调用 wttr.in 免费 API 获取实时天气（结果缓存 30 分钟） |
| `get_current_month` | — | 返回当前月份，格式 `YYYY-MM` |
| `fetch_external_data` | `user_id, month` | 查询指定用户指定月份的使用记录 |
| `get_user_location` | — | 返回当前用户所在城市（闭包注入） |
| `get_user_id` | — | 返回当前用户 ID（闭包注入） |
| `fill_context_for_report` | — | 触发中间件注入报告上下文，实现提示词切换 |

---

## 🧪 测试与自检

```bash
# 工具运行状态检查（语法 / 导入 / 工具调用 / 模型配置，共 4 层）
python verify_tools.py

# 会话持久化验证（含旧表迁移，不依赖模型与向量库）
python test_session_persist.py

# ReactAgent 集成验证（使用假模型，不消耗 API 额度）
python test_react_integration.py

# Streamlit 交互流程验证（空状态 / 历史加载 / 新对话）
python test_app_history.py
```

所有脚本均以退出码表示结果（`0` 通过 / `1` 失败），可直接接入 CI。

---

## ⚙️ 配置说明

<details>
<summary><b>config/rag.yml</b></summary>

```yaml
chat_model_name: qwen-max                       # 对话模型名称
embedding_model_name: text-embedding-v3         # Embedding 模型名称
api_base: ""                                    # API 端点（留空走环境变量或官方）
```

> `config/rag.yml.example` 为模板文件；含真实端点的本地配置建议放在 `config/rag.local.yml`（已 gitignore）。
</details>

<details>
<summary><b>config/chroma.yml</b></summary>

```yaml
collection_name: agent                          # 向量库集合名
persist_directory: chroma_db                    # 持久化目录
k: 3                                            # 检索返回条数
data_path: data                                 # 知识文档目录
md5_hex_store: md5.txt                          # 知识库同步标记（非权威，按实际结果重写）
allow_knowledge_file_type: [".txt", ".pdf"]     # 允许的文件类型（带不带点、大小写均可）
chunk_size: 200                                 # 分片长度
chunk_overlap: 20                               # 分片重叠
separators: ["\n\n", "。", ".", "?", "？", "!", " ", ""]
```
</details>

<details>
<summary><b>config/agent.yml</b></summary>

```yaml
external_data_path: data/external/records.csv   # 用户使用记录 CSV
session_db_path: data/sessions.db               # 会话数据库路径
```
</details>

<details>
<summary><b>config/prompts.yml</b></summary>

```yaml
main_prompt_path: prompts/main_prompt.txt           # 普通客服提示词
rag_summarize_prompt_path: prompts/rag_summarize.txt # RAG 总结提示词
report_prompt_path: prompts/report_prompt.txt       # 报告写手提示词
```
</details>

---

## 🔍 关键设计说明

### 报告模式切换机制

报告生成不是硬编码的分支逻辑，而是由 **模型自主判断 + 会话状态管理** 协同完成：

1. 提示词中强约束：判断用户意图为「生成报告」时，必须先调用 `fill_context_for_report`；
2. `ReactAgent` 检测到该工具名被调用 → `_switch_to_report_mode()` 把该会话的
   `session["mode"]` 置为 `report`（按 `conversation_id` 隔离）并落盘到 `conversations.mode`；
3. 下一轮循环组装消息时，`_get_system_prompt()` 按 `conversation_id` 读取模式并返回 **报告写手提示词**；
4. 报告产出后自动调用 `finish_report()` 恢复普通模式（`max_iterations` 用尽的收尾分支同样会复位，
   因此报告轮被中断也不会把模式留在 `report`）。

**模式状态的唯一权威来源是会话状态 `session["mode"]`**（持久化为 `conversations.mode`）。
中间件 `fill_context()` 仍会被调用，但只作按 `user_id` 的进程内存镜像、供其对外 HTTP 接口展示，
**不参与提示词切换决策**——否则进程重启后报告会话会读到过期镜像、用错提示词。

### 多会话模型

会话有两种状态：

- **active** —— 进行中，不出现在历史列表
- **archived** —— 已归档，出现在历史列表

```
打开网站  → archive_all_active() → create_conversation() → 空状态
点击新对话 → archive_conversation(当前) → create_conversation() → 空状态
点击历史   → archive_conversation(当前) → reactivate_conversation(目标) → 加载全部消息
```

空会话（无任何消息）在归档时直接删除，避免历史列表堆积无意义记录。
旧版单会话表（`sessions` / `messages`）在首次初始化时自动迁移为归档会话并删除旧表。

### 加载策略：启动预热 + 惰性兜底

实测（热盘）：`import rag.vector_store`（连带 chromadb 等）约 **8.7 秒**，RAG 服务的客户端与 embedding 构造约 **5.7 秒**，合计约 **15 秒**；冷盘首次可达 **40 秒**。`langchain_openai` 导入约 **7.5 秒**，ChatModel 创建约 **3-4 秒**。

**为什么知识库入库放在启动阶段**：Chroma 是单进程嵌入式向量库，写入与检索不能并发——一边 `add_documents` 一边 `retriever.invoke()` 会抛
`Error creating hnsw segment reader: Nothing found on disk`。因此把「写」收敛到启动时一次做完，运行期只「读」。对应到代码：

- `app.py` —— 启动时用 `@st.cache_resource` 执行一次 `warm_up_knowledge_base()` 完成入库与预热。
  **首次启动页面会停留约 15~40 秒**（界面有明确提示），之后每次 rerun 命中缓存瞬时返回；
  预热失败只记录日志并降级为「首次提问时懒加载」，不阻断页面
- `model/factory.py` —— `get_chat_model()` / `get_embed_model()` 首次调用时才创建
- `agent/tools/agents_tools.py` —— RAG 服务仍是延迟 import + 双检锁单例，作为预热失败时的兜底路径

---

## 📝 数据格式

`data/external/records.csv` 为合成测试数据，覆盖 10 个用户（`1001`–`1010`）× 12 个月（`2025-01` ~ `2025-12`）：

```csv
"用户ID","特征","清洁效率","耗材","对比","时间"
"1001","65㎡公寓 | 单身 | 木地板","覆盖率:85%\n日均清扫:45㎡\n漏扫区域:沙发底部（高度不足）","主刷寿命:剩余60天\nHEPA滤网:剩余40%","优于65%同面积用户（清洁频率更高）","2025-01"
```

> 字段内可用**字面 `\n`** 表示换行（写入提示词时保留为多行文本），字段本身不包含半角逗号。

---

## ⚠️ 注意事项

1. **API Key 安全**：密钥一律通过环境变量注入，不要写入配置文件。项目根目录的 `.env`（由 `.env.example` 复制而来）会在启动时自动加载，**已存在的系统环境变量优先、不会被覆盖**；同理，运行时以 `config/rag.yml` 为基线，若存在 `config/rag.local.yml` 则会覆盖同名字段，本地真实端点写在那里面即可（已被 `.gitignore` 忽略，仓库中仅保留 `config/rag.yml.example` 模板）。
2. **知识库增量同步**：应用启动（RAG 服务首次初始化）时会自动把 `data/` 下**新增 / 已修改 / 已删除**的知识文件同步进向量库——新增与修改的入库（修改的会先删掉旧分片，不会留下过期内容）、删除的清掉分片与 MD5 记录，**无需再手动执行** `python rag/vector_store.py`；该命令仍可用于单独预构建知识库、提前排除解析问题。同步按文件 MD5 判断，重复执行幂等。文件被清空或编码损坏导致解析不出内容时，也会清掉它的旧分片（否则等于"改了文件但改动不生效"，库里还在提供上一版内容）。
3. **天气工具**：依赖 `wttr.in` 公网服务，网络不通时返回兜底文案，不影响主流程。降雨概率取自该接口未来 6 小时的真实预报值，取不到时该段描述会被整体省略，不会用固定文案代替。
4. **外部数据热更新**：`data/external/records.csv` 按文件 `mtime` 判断是否需要重新加载，修改 CSV 后**无需重启进程**，下次调用自动生效。
5. **会话缓存**：`ReactAgent` 内存中的 `sessions` 与 SQLite 双写，进程重启后从 SQLite 恢复完整历史。会话库启用了 WAL 模式，多会话并发读写不会互相阻塞。
6. **报告模式退出**：用户触发报告生成后该会话进入报告模式；报告一旦产出即自动恢复普通模式，中途也可用侧边栏的「退出报告模式」手动结束。

---

## 📄 License

本项目仅用于学习与演示目的。

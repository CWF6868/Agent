"""
会话持久化存储（SQLite，多会话模型）

职责：
1. 以「会话（conversation）」为粒度持久化对话历史：一个用户可有多条会话记录
2. 会话状态：active（进行中，不出现在历史列表）/ archived（已归档，进入历史列表）
3. 页面流程配合：
   - 打开网站 → 归档上次未归档的会话 → 新建 active 会话（空状态）
   - 点击历史记录 → 加载该会话全部消息，可继续对话（自动转回 active）
4. 旧版单会话表（sessions/messages）自动迁移为归档会话

数据表：
  conversations (id, user_id, title, mode, status[active|archived], created_at, updated_at)
  conv_messages (id, conversation_id, seq, role, content, payload, created_at)
"""
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    title      TEXT NOT NULL DEFAULT '',
    mode       TEXT NOT NULL DEFAULT 'normal',
    status     TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conv_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    seq             INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL DEFAULT '',
    payload         TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, status, updated_at);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON conv_messages(conversation_id, seq);
"""

_TITLE_MAX_LEN = 20  # 会话标题取第一条用户消息的前 N 字


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _make_title(history: list) -> str:
    """从第一条用户消息生成会话标题"""
    for msg in history:
        if getattr(msg, "type", "") == "human":
            title = str(msg.content).strip().replace("\n", " ")
            return title[:_TITLE_MAX_LEN] if title else "新对话"
    return "新对话"


def _msg_to_row(conv_id: int, seq: int, msg) -> dict:
    """LangChain Message -> 存储行 dict（payload 序列化为 JSON 字符串）"""
    role = getattr(msg, "type", "human")
    content = msg.content if msg.content is not None else ""

    payload = {}
    if getattr(msg, "tool_calls", None):
        payload["tool_calls"] = msg.tool_calls
    if getattr(msg, "tool_call_id", None):
        payload["tool_call_id"] = msg.tool_call_id

    return {
        "conversation_id": conv_id,
        "seq": seq,
        "role": role,
        "content": content,
        "payload": json.dumps(payload, ensure_ascii=False),
        "created_at": _now(),
    }


def _row_to_msg(row: dict):
    """存储行 -> LangChain Message（重建可继续参与对话的消息对象）"""
    payload = json.loads(row["payload"] or "{}")
    if row["role"] == "ai":
        return AIMessage(content=row["content"], tool_calls=payload.get("tool_calls") or [])
    if row["role"] == "tool":
        return ToolMessage(content=row["content"], tool_call_id=payload.get("tool_call_id", ""))
    return HumanMessage(content=row["content"])


def _tc_id(tool_call: dict) -> str:
    """统一取 tool_call 的 id（缺失时归一为空串，便于做集合比较）"""
    return tool_call.get("id") or ""


def sanitize_history(history: list) -> list:
    """
    清洗恢复出来的历史，保证消息序列对模型 API 合法（幂等，不修改入参）。

    为什么需要它：落盘时 `AIMessage(tool_calls=...)` 与其配对的 `ToolMessage`
    分两次写入，进程若在两次写入之间中断（崩溃 / Streamlit 被 Stop / 强杀），
    磁盘上就会留下"声明了 N 个工具调用、却没有对应 tool 响应"的历史。
    这种历史原样发给模型会持续返回 400（assistant 消息声明的每个 tool_call
    都必须紧跟一条同 tool_call_id 的 tool 响应），而且**每次重试都会复现**，
    用户侧表现为该会话再也问不出任何回答。

    清洗规则：
    1. `AIMessage` 声明的 tool_call，若其后找不到配对的 tool 响应 → 从声明中剔除；
    2. 剔除后既无 tool_call 又无正文 → 整条丢弃（避免产生空 assistant 消息）；
    3. 找不到前置声明的"孤儿" tool 消息 → 丢弃（否则同样触发 400）。

    返回清洗后的新列表。
    """
    cleaned: list = []
    total = len(history)
    i = 0
    while i < total:
        msg = history[i]
        kind = getattr(msg, "type", "")

        # 孤儿 tool 消息：没有前置 AIMessage 声明（或声明已被剔除）→ 丢弃
        if kind == "tool":
            i += 1
            continue

        tool_calls = list(getattr(msg, "tool_calls", None) or []) if kind == "ai" else []
        if not tool_calls:
            cleaned.append(msg)
            i += 1
            continue

        # 收集紧随其后的连续 tool 响应
        j = i + 1
        followers = []
        while j < total and getattr(history[j], "type", "") == "tool":
            followers.append(history[j])
            j += 1

        answered = {getattr(m, "tool_call_id", "") or "" for m in followers}
        paired = [tc for tc in tool_calls if _tc_id(tc) in answered]

        if not paired:
            # 全部调用都没有响应：正文有内容则退化为纯文本 assistant 消息，否则整条丢弃
            if str(msg.content or "").strip():
                cleaned.append(AIMessage(content=msg.content))
        elif len(paired) == len(tool_calls):
            # 全部配对成功：原样保留
            cleaned.append(msg)
            cleaned.extend(followers)
        else:
            # 部分配对成功（多工具调用执行到一半被打断）
            cleaned.append(AIMessage(content=msg.content, tool_calls=paired))
            paired_ids = {_tc_id(tc) for tc in paired}
            cleaned.extend(
                m for m in followers if (getattr(m, "tool_call_id", "") or "") in paired_ids
            )
        i = j

    return cleaned


class SessionStore:
    """
    线程安全的 SQLite 多会话存储。

    用法：
        store = SessionStore()
        conv_id = store.create_conversation("1001")        # 新建 active 会话
        store.save_conversation(conv_id, "normal", history) # 全量写回消息
        store.archive_all_active()                         # 打开网站时归档未完成会话（全部用户）
        store.list_archived("1001")                        # 历史会话列表
        ui = store.build_ui_messages(conv_id)              # 页面展示数据
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            # 环境变量覆盖仅供测试使用：让页面级验证（Streamlit AppTest 会真实地
            # 新建/归档会话）指向临时库，避免污染正在使用的 data/sessions.db。
            # 生产运行时该变量不存在，走 config/agent.yml 的 session_db_path。
            db_path = os.getenv("AGENT_SESSION_DB") or get_abs_path(
                agent_conf.get("session_db_path", "data/sessions.db")
            )
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    # ---------- 基础连接 ----------

    def _connect(self):
        """创建连接并应用并发相关设置。

        - WAL：读不阻塞写、写不阻塞读，多会话并发时不再互相顶掉
        - busy_timeout：遇到锁时等待而非立刻抛 database is locked
        - synchronous=NORMAL：WAL 模式下的推荐档位，兼顾安全与写入速度
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @contextmanager
    def _connection(self):
        """连接作用域：正常退出提交、异常回滚、无论如何都关闭连接。

        原先直接 `with self._connect() as conn` 只负责提交/回滚事务，
        连接本身要等 GC 回收；显式 close 避免文件句柄在长跑进程中堆积。
        """
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """建表 + 旧版数据迁移（幂等）"""
        with self._lock, self._connection() as conn:
            conn.executescript(_SCHEMA)
            self._migrate_legacy(conn)
            conn.commit()

    def _migrate_legacy(self, conn):
        """
        迁移旧版单会话表（sessions/messages）为归档会话。
        每个旧用户生成一条 archived 会话，标题取第一条用户消息。
        """
        tables = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "sessions" not in tables or "messages" not in tables:
            return

        conv_count = conn.execute(
            "SELECT COUNT(*) AS c FROM conversations"
        ).fetchone()["c"]
        if conv_count == 0:
            old_sessions = conn.execute("SELECT user_id FROM sessions").fetchall()
            for s in old_sessions:
                uid = s["user_id"]
                old_msgs = conn.execute(
                    "SELECT role, content, payload FROM messages WHERE user_id=? ORDER BY id",
                    (uid,),
                ).fetchall()
                if not old_msgs:
                    continue
                now = _now()
                title = "旧会话"
                for m in old_msgs:
                    if m["role"] == "human":
                        title = m["content"].strip().replace("\n", " ")[:_TITLE_MAX_LEN] or "旧会话"
                        break
                cur = conn.execute(
                    "INSERT INTO conversations(user_id, title, mode, status, created_at, updated_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (uid, title, "normal", "archived", now, now),
                )
                conv_id = cur.lastrowid
                conn.executemany(
                    "INSERT INTO conv_messages(conversation_id, seq, role, content, payload, created_at) "
                    "VALUES(?,?,?,?,?,?)",
                    [
                        (conv_id, i, m["role"], m["content"], m["payload"], now)
                        for i, m in enumerate(old_msgs)
                    ],
                )
        # 迁移完成后删除旧表
        conn.execute("DROP TABLE IF EXISTS messages")
        conn.execute("DROP TABLE IF EXISTS sessions")

    # ---------- 会话生命周期 ----------

    def create_conversation(self, user_id: str) -> int:
        """新建 active 会话，返回会话 ID"""
        now = _now()
        with self._lock, self._connection() as conn:
            cur = conn.execute(
                "INSERT INTO conversations(user_id, title, mode, status, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?)",
                (user_id, "", "normal", "active", now, now),
            )
            conn.commit()
            return cur.lastrowid

    def archive_all_active(self, user_id: str | None = None) -> int:
        """归档 active 会话（打开网站/切换用户/新建会话时调用），返回处理数量。
        空会话（无消息）直接删除，避免历史列表出现无意义记录。

        user_id 为 None 时归档**全部用户**的 active 会话，这是 app 的默认调用方式。
        原因：active 只代表"当前正在对话的那一条"，而历史列表只查 archived
        （list_archived），因此任何被遗弃的 active —— 例如 A 用户切到 B 用户后，
        A 的会话仍是 active 且属主已不在当前筛选范围内 —— 都会变成既不出现在
        历史列表、也不会被清理的孤儿会话，其对话内容永久不可见。
        全局收起可维持"同一时刻至多一条 active"这一不变量，从根上消除孤儿。
        """
        with self._lock, self._connection() as conn:
            if user_id is None:
                rows = conn.execute(
                    "SELECT id FROM conversations WHERE status='active'"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id FROM conversations WHERE user_id=? AND status='active'",
                    (user_id,),
                ).fetchall()
            for r in rows:
                self._archive_row(conn, r["id"])
            conn.commit()
            return len(rows)

    def archive_conversation(self, conv_id: int):
        """归档单个会话（点击新对话时调用）；空会话直接删除"""
        with self._lock, self._connection() as conn:
            self._archive_row(conn, conv_id)
            conn.commit()

    def _archive_row(self, conn, conv_id: int):
        """归档单行：有消息置 archived，空会话直接删除"""
        cnt = conn.execute(
            "SELECT COUNT(*) AS c FROM conv_messages WHERE conversation_id=?",
            (conv_id,),
        ).fetchone()["c"]
        if cnt == 0:
            conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
        else:
            conn.execute(
                "UPDATE conversations SET status='archived', updated_at=? WHERE id=?",
                (_now(), conv_id),
            )

    def reactivate_conversation(self, conv_id: int):
        """将归档会话转回 active（从历史列表选中继续对话时调用）"""
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE conversations SET status='active', updated_at=? WHERE id=?",
                (_now(), conv_id),
            )
            conn.commit()

    def delete_conversation(self, conv_id: int):
        """删除会话及其全部消息"""
        with self._lock, self._connection() as conn:
            conn.execute("DELETE FROM conv_messages WHERE conversation_id=?", (conv_id,))
            conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
            conn.commit()

    def clear_user(self, user_id: str):
        """删除某用户的全部会话（含消息）"""
        with self._lock, self._connection() as conn:
            conn.execute(
                "DELETE FROM conv_messages WHERE conversation_id IN "
                "(SELECT id FROM conversations WHERE user_id=?)",
                (user_id,),
            )
            conn.execute("DELETE FROM conversations WHERE user_id=?", (user_id,))
            conn.commit()

    # ---------- 会话读写 ----------

    def get_conversation(self, conv_id: int) -> dict:
        """恢复单个会话（含完整历史）；不存在时返回 None"""
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id=?", (conv_id,)
            ).fetchone()
            if row is None:
                return None
            rows = conn.execute(
                "SELECT * FROM conv_messages WHERE conversation_id=? ORDER BY seq",
                (conv_id,),
            ).fetchall()
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "title": row["title"],
            "mode": row["mode"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            # 恢复时统一清洗：历史上可能残留"声明了 tool_calls 却缺配对响应"的
            # 记录（写入途中断所致），不清洗会让该会话之后每次调用模型都 400
            "history": sanitize_history([_row_to_msg(r) for r in rows]),
        }

    def append_messages(self, conv_id: int, mode: str, new_messages: list) -> int:
        """
        增量追加消息（不重写既有行），并同步 mode / title / status / updated_at。

        与 save_conversation 的区别：save_conversation 每次调用都先 DELETE 全部
        消息再全量 INSERT。一轮对话会触发 4~6 次写入（用户消息、每轮 AI+工具响应、
        最终回答、报告模式切换），单次成本又是 O(历史长度)，合起来是 O(n²) 的
        磁盘写放大 —— 对话一长就明显变慢。而历史只会在**末尾追加**，所以只需写入
        尚未落盘的那一段：seq 从既有 MAX(seq)+1 接续，单次写放大降到"本次新增条数"。

        new_messages 为空时只做状态同步（例如仅切换了 mode），不产生 INSERT。

        Args:
            conv_id: 会话 ID
            mode: 当前模式（normal | report）
            new_messages: 本次新增的 LangChain Message 列表（按时间顺序）

        Returns:
            实际写入的消息条数；会话已被删除时返回 0。
        """
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT title FROM conversations WHERE id=?", (conv_id,)
            ).fetchone()
            if row is None:
                return 0  # 会话已被删除，忽略写入
            title = row["title"] or _make_title(new_messages)

            if new_messages:
                base = conn.execute(
                    "SELECT COALESCE(MAX(seq), -1) FROM conv_messages WHERE conversation_id=?",
                    (conv_id,),
                ).fetchone()[0]
                conn.executemany(
                    "INSERT INTO conv_messages(conversation_id, seq, role, content, payload, created_at) "
                    "VALUES(:conversation_id, :seq, :role, :content, :payload, :created_at)",
                    [
                        _msg_to_row(conv_id, base + 1 + i, m)
                        for i, m in enumerate(new_messages)
                    ],
                )

            conn.execute(
                "UPDATE conversations SET title=?, mode=?, status='active', updated_at=? WHERE id=?",
                (title, mode, _now(), conv_id),
            )
            conn.commit()
            return len(new_messages)

    def save_conversation(self, conv_id: int, mode: str, history: list) -> None:
        """全量写回会话（消息先清后插；首次对话时自动生成标题）

        仅用于需要**覆盖**既有历史的场景（测试预置数据、外部调用方）。
        增量追加请用 append_messages()。
        """
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT user_id, title FROM conversations WHERE id=?", (conv_id,)
            ).fetchone()
            if row is None:
                return  # 会话已被删除，忽略写入
            title = row["title"] or _make_title(history)

            conn.execute("DELETE FROM conv_messages WHERE conversation_id=?", (conv_id,))
            conn.execute(
                "UPDATE conversations SET title=?, mode=?, status='active', updated_at=? WHERE id=?",
                (title, mode, _now(), conv_id),
            )
            conn.executemany(
                "INSERT INTO conv_messages(conversation_id, seq, role, content, payload, created_at) "
                "VALUES(:conversation_id, :seq, :role, :content, :payload, :created_at)",
                [_msg_to_row(conv_id, i, m) for i, m in enumerate(history)],
            )
            conn.commit()

    # ---------- 列表与 UI 数据 ----------

    def list_archived(self, user_id: str) -> list[dict]:
        """返回某用户的历史（archived）会话，按更新时间倒序"""
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT c.id, c.title, c.created_at, c.updated_at, "
                "COUNT(m.id) AS message_count "
                "FROM conversations c "
                "LEFT JOIN conv_messages m ON c.id = m.conversation_id "
                "WHERE c.user_id=? AND c.status='archived' "
                "GROUP BY c.id ORDER BY c.updated_at DESC",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def build_ui_messages(self, conv_id: int) -> list[dict]:
        """
        恢复页面展示用的消息列表。
        Returns:
            [{"role": "user"|"assistant", "content": str,
              "tool_calls": [{"name","args","result","call_id"}]}]
        """
        conv = self.get_conversation(conv_id)
        if conv is None:
            return []
        ui = []
        pending = []  # 当前 assistant 消息声明的工具调用，等待 tool 消息填充结果

        for msg in conv["history"]:
            if msg.type == "human":
                ui.append({"role": "user", "content": msg.content})
            elif msg.type == "ai":
                calls = [
                    {
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {}),
                        "result": "",
                        "call_id": tc.get("id", ""),
                    }
                    for tc in (msg.tool_calls or [])
                ]
                ui.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": calls,
                })
                pending = calls
            elif msg.type == "tool":
                for c in pending:
                    if c["call_id"] == msg.tool_call_id:
                        c["result"] = str(msg.content)
                        break
        return ui

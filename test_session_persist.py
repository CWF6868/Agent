# -*- coding: utf-8 -*-
"""
会话持久化功能验证脚本（多会话模型，不依赖模型 / 不依赖 RAG 向量库）
运行：python test_session_persist.py [db_path]
"""
import os
import sqlite3
import sys
import tempfile

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.tools.session_store import SessionStore  # noqa: E402

FAIL = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAIL.append(name)


def build_history():
    return [
        HumanMessage(content="生成我的使用报告"),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "fill_context_for_report",
                "args": {},
                "id": "call_ctx_1",
                "type": "tool_call",
            }],
        ),
        ToolMessage(content="fill_context_for_report已调用", tool_call_id="call_ctx_1"),
        AIMessage(content="正在为您生成使用报告..."),
    ]


def main():
    db_dir = tempfile.mkdtemp(prefix="session_test_")
    db_path = os.path.join(db_dir, "sessions.db")

    # 1) 新建 store、创建会话并保存
    store = SessionStore(db_path=db_path)
    conv_id = store.create_conversation("1001")
    check("create_conversation 返回数字 ID", isinstance(conv_id, int))

    history = build_history()
    store.save_conversation(conv_id, "report", history)
    check("save 后自动生成标题", store.get_conversation(conv_id)["title"] == "生成我的使用报告",
          f"实际 {store.get_conversation(conv_id)['title']!r}")

    # 2) 会话状态与列表
    conv = store.get_conversation(conv_id)
    check("消息数为 4", len(conv["history"]) == 4, f"实际 {len(conv['history'])}")
    check("mode 为 report", conv["mode"] == "report")
    check("active 会话不出现在历史列表", store.list_archived("1001") == [])

    store.archive_conversation(conv_id)
    archived = store.list_archived("1001")
    check("归档后出现在历史列表", len(archived) == 1)
    check("历史列表消息数为 4", archived[0]["message_count"] == 4, f"实际 {archived[0]['message_count']}")

    # 3) 模拟进程重启：新 store 实例从同一 db 恢复
    store2 = SessionStore(db_path=db_path)
    restored = store2.get_conversation(conv_id)
    check("重启后消息数为 4", len(restored["history"]) == 4)
    h = restored["history"]
    check("第 2 条为 AIMessage 且 tool_calls 恢复",
          isinstance(h[1], AIMessage) and h[1].tool_calls[0]["name"] == "fill_context_for_report")
    check("第 3 条 ToolMessage tool_call_id 恢复",
          isinstance(h[2], ToolMessage) and h[2].tool_call_id == "call_ctx_1")

    # 4) UI 展示数据（工具调用结果配对）
    ui = store2.build_ui_messages(conv_id)
    check("UI 消息数为 3", len(ui) == 3, f"实际 {len(ui)}")
    check("工具调用结果已配对", ui[1]["tool_calls"][0]["result"] == "fill_context_for_report已调用",
          f"实际 {ui[1]['tool_calls'][0]['result']}")

    # 5) reactivate / archive 循环
    store2.reactivate_conversation(conv_id)
    check("reactivate 后不在历史列表", store2.list_archived("1001") == [])
    store2.archive_conversation(conv_id)
    check("再次归档后回到历史列表", len(store2.list_archived("1001")) == 1)

    # 6) 多会话并存：同一用户两个会话
    conv2 = store2.create_conversation("1001")
    store2.save_conversation(conv2, "normal", [HumanMessage(content="扫地机器人怎么保养？")])
    store2.archive_conversation(conv2)
    check("同一用户两条历史记录", len(store2.list_archived("1001")) == 2)
    check("另一用户看不到这些记录", store2.list_archived("2002") == [])

    # 7) 删除会话
    store2.delete_conversation(conv2)
    check("删除后历史列表剩 1 条", len(store2.list_archived("1001")) == 1)
    check("删除后 get_conversation 返回 None", store2.get_conversation(conv2) is None)

    # 8) 未知会话
    check("未知会话返回 None", store.get_conversation(999999) is None)
    check("未知会话 UI 为空", store.build_ui_messages(999999) == [])

    # 9) 旧版表结构自动迁移
    legacy_dir = tempfile.mkdtemp(prefix="legacy_test_")
    legacy_path = os.path.join(legacy_dir, "legacy.db")
    conn = sqlite3.connect(legacy_path)
    conn.executescript("""
        CREATE TABLE sessions (user_id TEXT PRIMARY KEY, mode TEXT, updated_at TEXT);
        CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT,
                               seq INTEGER, role TEXT, content TEXT, payload TEXT, created_at TEXT);
        INSERT INTO sessions VALUES ('9001', 'normal', '2026-01-01');
        INSERT INTO messages (user_id, seq, role, content, payload, created_at) VALUES
            ('9001', 0, 'human', '旧对话第一句', '{}', '2026-01-01');
    """)
    conn.commit()
    conn.close()

    legacy_store = SessionStore(db_path=legacy_path)
    check("旧表已删除", not any(
        r[0] in ("sessions", "messages")
        for r in sqlite3.connect(legacy_path).execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ))
    legacy_list = legacy_store.list_archived("9001")
    check("旧数据迁移为 1 条归档会话", len(legacy_list) == 1)
    check("迁移会话标题取首条用户消息", legacy_list[0]["title"] == "旧对话第一句",
          f"实际 {legacy_list[0]['title']!r}")
    legacy_conv = legacy_store.get_conversation(legacy_list[0]["id"])
    check("迁移消息内容保留", legacy_conv["history"][0].content == "旧对话第一句")

    print("-" * 40)
    if FAIL:
        print(f"共 {len(FAIL)} 项失败: {FAIL}")
        sys.exit(1)
    print("全部测试通过")


if __name__ == "__main__":
    main()

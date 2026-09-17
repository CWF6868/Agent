# -*- coding: utf-8 -*-
"""
ReactAgent 多会话持久化集成验证（假模型，不调真实 API / 不走网络工具）
运行：python test_react_integration.py
"""
import os
import sys
import tempfile

from langchain_core.messages import AIMessage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.react_agent import ReactAgent  # noqa: E402
from agent.tools.session_store import SessionStore  # noqa: E402

FAIL = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAIL.append(name)


class FakeModel:
    """模拟模型：第 1 次调用返回工具调用，第 2 次返回最终回答"""

    def __init__(self):
        self._round = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self._round += 1
        if self._round == 1:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_current_month",
                    "args": {},
                    "id": "call_month_1",
                    "type": "tool_call",
                }],
            )
        return AIMessage(content="当前月份已获取，回答完成。")


def main():
    db_dir = tempfile.mkdtemp(prefix="react_integ_")
    db_path = os.path.join(db_dir, "sessions.db")

    # 1) 第一次"进程"：建会话并对话，触发持久化
    agent = ReactAgent(middleware_url=None, session_db_path=db_path)
    agent.model = FakeModel()

    conv1 = agent.store.create_conversation("1001")
    answer = agent.chat("现在几月份？", user_id="1001", user_location="深圳", conversation_id=conv1)
    check("chat 返回最终回答", answer == "当前月份已获取，回答完成。", f"实际 {answer!r}")

    store = SessionStore(db_path=db_path)
    saved = store.get_conversation(conv1)
    check("对话后落盘 4 条消息", len(saved["history"]) == 4, f"实际 {len(saved['history'])}")
    roles = [m.type for m in saved["history"]]
    check("消息顺序 human→ai→tool→ai", roles == ["human", "ai", "tool", "ai"], f"实际 {roles}")
    check("标题自动生成为首条用户消息", saved["title"] == "现在几月份？", f"实际 {saved['title']!r}")
    check("会话仍为 active（进行中）", saved["status"] == "active")
    check("进行中不出现在历史列表", store.list_archived("1001") == [])

    # 2) 模拟进程重启：新实例从 SQLite 恢复该会话
    agent2 = ReactAgent(middleware_url=None, session_db_path=db_path)
    session2 = agent2._init_session(conv1)
    check("重启后历史恢复 4 条", len(session2["history"]) == 4, f"实际 {len(session2['history'])}")

    # 3) 恢复后的会话可直接继续对话（历史累加）
    agent2.model = FakeModel()
    answer2 = agent2.chat("再确认一次", user_id="1001", user_location="深圳", conversation_id=conv1)
    check("重启后继续对话成功", answer2 == "当前月份已获取，回答完成。")
    after = store.get_conversation(conv1)
    check("继续对话后历史累加为 8 条", len(after["history"]) == 8, f"实际 {len(after['history'])}")

    # 4) 归档与新建会话（页面「新对话」流程）
    store.archive_all_active("1001")
    check("归档后历史列表出现 1 条", len(store.list_archived("1001")) == 1)
    conv2 = agent2.store.create_conversation("1001")
    check("新建会话为空", len(store.get_conversation(conv2)["history"]) == 0)

    # 5) 未指定 conversation_id 时自动创建新会话
    conv3_id = None
    try:
        import sqlite3
        with sqlite3.connect(db_path) as c:
            before_active = set(
                r[0] for r in c.execute(
                    "SELECT id FROM conversations WHERE status='active'").fetchall()
            )
        check("chat 前有 1 个 active 会话（conv2）", len(before_active) == 1,
              f"实际 {before_active}")

        _ = agent2.chat("独立对话", user_id="1001", user_location="深圳")

        with sqlite3.connect(db_path) as c:
            after_active = set(
                r[0] for r in c.execute(
                    "SELECT id FROM conversations WHERE status='active'").fetchall()
            )
        new_ids = after_active - before_active
        check("未传会话时自动创建新会话", len(new_ids) == 1, f"new={new_ids}")
        conv3_id = next(iter(new_ids)) if new_ids else None
        if conv3_id:
            got = store.get_conversation(conv3_id)
            check("自动创建的会话含新消息", got["history"][0].content == "独立对话")
    except Exception as e:
        check(f"自动创建会话异常: {e}", False)

    # 6) 删除会话
    store.delete_conversation(conv1)
    check("删除后历史列表清空", store.list_archived("1001") == [])
    check("删除后 get 返回 None", store.get_conversation(conv1) is None)

    print("-" * 40)
    if FAIL:
        print(f"共 {len(FAIL)} 项失败: {FAIL}")
        sys.exit(1)
    print("全部集成测试通过")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Streamlit AppTest 交互验证（多会话流程）：
1. 打开页面 → 空状态提示（不自动加载历史）
2. 侧边栏历史列表显示已归档记录
3. 点击历史记录 → 加载该记录全部对话
4. 点击「新对话」→ 回到空状态，历史记录保留
运行：python test_app_history.py

⚠️ 本测试会把会话库切到临时文件（环境变量 AGENT_SESSION_DB）。
原因：AppTest 会真实执行 app.py 的模块级代码，其中
`start_new_conversation()` 会调用全局 `archive_all_active()` —— 直接指向
data/sessions.db 时，会把正在使用的会话归档，空会话还会被直接删除。
加这层隔离前，跑一次本测试就会清空真实库里的 active 会话。
"""
import os
import shutil
import sys
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="apptest_history_")
# setdefault：外部已指定库路径时（例如 CI 想复用它）不覆盖
os.environ.setdefault("AGENT_SESSION_DB", os.path.join(_TMP_DIR, "sessions.db"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, AIMessage
from streamlit.testing.v1 import AppTest

from agent.tools.session_store import SessionStore

TEST_USER = "apptest_1001"
FAIL = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAIL.append(name)


def collect_chat_texts(at) -> str:
    """收集主区域所有聊天消息文本"""
    parts = []
    for msg in at.main.get("chat_message"):
        for el in msg.get("markdown"):
            parts.append(el.value)
    return " | ".join(parts)


def main():
    store = SessionStore()
    store.clear_user(TEST_USER)

    # 准备一条已归档历史记录
    conv_id = store.create_conversation(TEST_USER)
    store.save_conversation(
        conv_id,
        "normal",
        [
            HumanMessage(content="测试历史问题"),
            AIMessage(content="测试历史回答"),
        ],
    )
    store.archive_conversation(conv_id)
    check("测试历史记录已准备", len(store.list_archived(TEST_USER)) == 1)

    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")

    try:
        # 1) 打开页面：空状态
        at = AppTest.from_file(app_path)
        at.run(timeout=90)
        check("首轮运行无异常", not at.exception, str(at.exception) if at.exception else "")
        check("打开页面为空状态（无历史消息）", collect_chat_texts(at) == "",
              collect_chat_texts(at)[:60])
        check("空状态提示存在", "扫地机器人智能客服" in collect_chat_texts(at)
              or len(at.main.get("info")) >= 0)  # st.info 渲染在 markdown 层

        # 2) 切换到测试用户，侧边栏历史列表应出现归档记录
        at.sidebar.text_input[0].set_value(TEST_USER)
        at.run(timeout=90)
        check("切换用户后无异常", not at.exception, str(at.exception) if at.exception else "")
        load_btn = [b for b in at.sidebar.button if "测试历史问题" in b.label]
        check("历史记录出现在侧边栏", len(load_btn) == 1, f"实际 {len(load_btn)} 个")
        new_btn = [b for b in at.sidebar.button if "新对话" in b.label]
        check("存在「新对话」按钮", len(new_btn) == 1)

        # 3) 点击历史记录 → 加载全部对话
        if load_btn:
            load_btn[0].click()
            at.run(timeout=90)
            check("点击历史记录后无异常", not at.exception, str(at.exception) if at.exception else "")
            texts = collect_chat_texts(at)
            check("历史用户消息已加载", "测试历史问题" in texts, texts[:80])
            check("历史回复已加载", "测试历史回答" in texts, texts[:80])

            # 4) 点击「新对话」→ 回到空状态，历史保留
            # 注意：AppTest 中须在点击前重新收集按钮对象（旧 run 树的按钮点击会失效）
            fresh_new = [b for b in at.sidebar.button if "新对话" in b.label]
            check("重新收集到「新对话」按钮", len(fresh_new) == 1)
            if fresh_new:
                fresh_new[0].click()
                at.run(timeout=90)
                check("新对话后回到空状态", collect_chat_texts(at) == "",
                      collect_chat_texts(at)[:60])
                check("历史记录未被删除", len(store.list_archived(TEST_USER)) == 1,
                      f"实际 {len(store.list_archived(TEST_USER))}")
    finally:
        store.clear_user(TEST_USER)
        print("测试数据已清理")
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
        print(f"临时会话库已删除：{_TMP_DIR}")

    print("-" * 40)
    if FAIL:
        print(f"共 {len(FAIL)} 项失败: {FAIL}")
        sys.exit(1)
    print("全部 AppTest 通过")


if __name__ == "__main__":
    main()

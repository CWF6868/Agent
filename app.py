"""
扫地机器人智能客服 Web 界面（基于 Streamlit）

运行方式：
    cd D:\Agent
    streamlit run app.py

功能：
    - 打开网站始终显示空状态，每次对话为一条独立会话记录（多会话模型）
    - 对话实时保存；点击「新对话」把当前会话归档为历史记录，并开始新会话
    - 历史会话列表（按当前用户过滤）：选中可查看全部对话并继续；可删除
    - 重启网站后：未归档的会话自动归档，历史记录随时可从列表恢复
"""
import streamlit as st
from datetime import datetime

from agent.react_agent import ReactAgent
from agent.tools import middleware
from agent.tools.session_store import SessionStore
from utils.logger_handler import logger

# 会话持久化存储（SQLite）：历史会话列表、消息恢复都从这读取
session_store = SessionStore()


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="扫地机器人智能客服",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 历史会话切换处理（必须在 user_id 输入框实例化之前执行）
# ============================================================
# Streamlit 不允许在 widget 实例化后再修改其 session_state 值，
# 因此点击历史会话时只写入 pending_switch_user（普通状态键），
# 在下一轮 rerun 的最开头、user_id 输入框尚未实例化时再写入其 key。
# 默认用户也在这一阶段统一初始化，避免与 text_input 的 value 参数冲突。
if "user_id_input" not in st.session_state:
    st.session_state["user_id_input"] = "1001"

if "pending_switch_user" in st.session_state:
    st.session_state["user_id_input"] = st.session_state.pop("pending_switch_user")


# ============================================================
# Agent 实例缓存（避免每次 rerun 重新加载 RAG 向量库）
# ============================================================
@st.cache_resource
def get_agent(middleware_url: str) -> ReactAgent:
    """
    根据中间件地址创建并缓存 ReactAgent 实例。
    middleware_url 为 None 时使用本地函数模式。
    相同的 middleware_url 只会初始化一次。
    """
    return ReactAgent(middleware_url=middleware_url)


# ============================================================
# 知识库预热：把「写入向量库」挪出用户请求路径
# ============================================================
# 入库是写 Chroma 的操作，而 Chroma 是单进程嵌入式向量库，读写并发会让
# hnsw 索引读取失败（Error creating hnsw segment reader: Nothing found on disk）。
# 因此把首次全量入库放在应用启动阶段完成（st.cache_resource 保证同进程仅执行一次），
# 用户提问时只读向量库。预热失败不阻断页面，提问时仍会走懒加载兜底。
@st.cache_resource
def warm_up_knowledge_base() -> bool:
    try:
        from rag.rag_service import RagSummarizeService
        RagSummarizeService()
        logger.info("[app] 知识库预热完成")
        return True
    except Exception as e:
        logger.error(f"[app] 知识库预热失败，改为首次提问时懒加载：{e}", exc_info=True)
        return False


with st.spinner("正在加载知识库..."):
    warm_up_knowledge_base()


def start_new_conversation(user_id: str):
    """
    开始新会话（空状态）：
    1. 归档**所有**进行中的会话（含其它用户遗留的），再新建当前用户的空会话。
       这里刻意不按 user_id 过滤：active 只代表"当前正在对话的那一条"，
       而历史列表只显示 archived，所以被遗弃的 active（切换用户、
       浏览器中途关闭）会变成既看不见也清不掉的孤儿会话。全局收起后，
       "同一时刻至多一条 active" 成为不变量。
    2. 新建一个 active 会话作为当前会话
       （新会话记录默认 mode='normal'；模式按 conversation_id 隔离，
        报告模式不会被带到新会话）
    3. 同步清理中间件上下文镜像（模式权威源是会话状态，这里只是保持镜像一致）
    """
    session_store.archive_all_active()
    conv_id = session_store.create_conversation(user_id)
    middleware.clear_context(user_id)
    st.session_state.current_conv_id = conv_id
    st.session_state.messages = []
    st.session_state.current_conv_user = user_id


def fmt_time(ts: str) -> str:
    """ISO 时间戳 -> MM-DD HH:mm"""
    try:
        return datetime.fromisoformat(ts).strftime("%m-%d %H:%M")
    except Exception:
        return ts


# ============================================================
# 侧边栏：配置面板 + 新对话 + 历史会话
# ============================================================
with st.sidebar:
    st.header("⚙️ 系统配置")

    user_id = st.text_input("用户 ID", key="user_id_input", help="用于隔离不同用户的会话和使用记录")
    user_location = st.text_input("所在城市", value="深圳", help="天气查询和地域适配使用")

    st.divider()
    st.subheader("中间件设置")
    middleware_mode = st.radio(
        "运行模式",
        ["本地函数", "HTTP 服务"],
        help="本地函数：agent 与中间件同进程；HTTP 服务：需先运行 python middleware.py",
    )
    if middleware_mode == "HTTP 服务":
        middleware_url = st.text_input("中间件地址", value="http://127.0.0.1:8000")
    else:
        middleware_url = None

    st.divider()

    # 当前模式指示器（读会话自身状态）
    # session["mode"] 是模式的唯一权威源，按 conversation_id 存储并持久化在
    # conversations.mode；不再读中间件按 user_id 维护的进程内存上下文，
    # 否则进程重启后指示器会与真实状态脱节，同一用户多会话也会互相干扰。
    _cid = st.session_state.get("current_conv_id")
    # 切换用户时 current_conv_id 仍指向上一个用户的会话，此时按即将新建的普通会话展示
    _conv = (
        session_store.get_conversation(_cid)
        if _cid and st.session_state.get("current_conv_user") == user_id
        else None
    )
    current_mode = _conv["mode"] if _conv else "normal"
    if current_mode == "report":
        st.markdown("### 📊 当前：报告模式")
        st.caption("已注入报告上下文，使用报告提示词")
        # 手动退出通道：报告已输出后 agent 会自动退出，这里用于中途反悔的情况
        if st.button("↩️ 退出报告模式", use_container_width=True):
            get_agent(middleware_url).finish_report(user_id, _cid)
            st.rerun()
    else:
        st.markdown("### 💬 当前：普通模式")
        st.caption("标准客服对话模式")

    # 状态自检（调试用）：把「权威模式」与「中间件镜像」并排show出来。
    # 会话模式（按 conversation_id，落盘在 conversations.mode）是唯一权威源，
    # 决定下一轮用哪套提示词；中间件上下文（按 user_id，存进程内存）只是镜像，
    # 进程重启或切换历史会话后可能与权威源不一致——有分歧时以权威模式为准。
    with st.expander("🔍 状态自检（调试）", expanded=False):
        _mirror = middleware.get_context(user_id).get("mode", "normal")
        if _conv:
            _auth = _conv["mode"]
            st.markdown(f"- 会话 ID：`{_conv['id']}`（{_conv['status']}）")
            st.markdown(f"- 权威模式（会话，conversations.mode）：`{_auth}` ← 决定本轮提示词")
            st.markdown(f"- 中间件镜像（user_id，进程内存）：`{_mirror}`")
            if _auth != _mirror:
                st.warning("两源不一致：以权威模式为准，镜像已过期（进程重启后就会出现这种情况）")
            else:
                st.caption("两源一致")
            st.caption("提示词应为：报告 prompt" if _auth == "report" else "提示词应为：普通客服 prompt")
        else:
            st.caption("尚无当前会话，进入对话后显示")

    st.divider()

    # 新对话按钮：归档当前会话 → 开始新的空会话（历史记录不会被删除）
    # 按钮点击后 Streamlit 会自动 rerun，无需手动调用
    if st.button("🆕 新对话", use_container_width=True, type="secondary"):
        start_new_conversation(user_id)

    st.divider()

    # 历史会话列表（仅显示已归档会话，进行中的会话不出现在这里）
    st.subheader("📚 历史会话")
    history_list = session_store.list_archived(user_id)
    if history_list:
        for conv in history_list:
            title = conv["title"] or "新对话"
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(
                    f"{title}　{fmt_time(conv['updated_at'])} · {conv['message_count']} 条",
                    key=f"load_{conv['id']}",
                    use_container_width=True,
                ):
                    # 切换前先归档当前会话（若与目标不同），避免多个 active 并存
                    cur = st.session_state.get("current_conv_id")
                    if cur and cur != conv["id"]:
                        session_store.archive_conversation(cur)
                    # 上一个会话已结束，其报告模式不应带到这条历史会话中
                    middleware.clear_context(user_id)
                    # 目标记录转 active：加载全部对话，可继续对话
                    session_store.reactivate_conversation(conv["id"])
                    st.session_state.current_conv_id = conv["id"]
                    st.session_state.messages = session_store.build_ui_messages(conv["id"])
                    st.session_state.current_conv_user = user_id
            with col2:
                with st.popover("🗑"):
                    st.caption("删除这条历史记录？")
                    if st.button("确认删除", key=f"del_{conv['id']}", use_container_width=True):
                        session_store.delete_conversation(conv["id"])
    else:
        st.caption("暂无历史会话，对话后点击「新对话」自动保存")

    st.caption(f"中间件模式：{middleware_mode}")


# ============================================================
# 主区域：标题 + 会话初始化 + 对话界面
# ============================================================
st.title("🤖 扫地机器人智能客服")
st.caption("ReAct 架构 · 支持工具调用 · 支持个人使用报告生成")

# 首次进入 / 切换用户：归档未完成会话并新建空会话（每次打开网站都是空状态）
if st.session_state.get("current_conv_user") != user_id or "current_conv_id" not in st.session_state:
    start_new_conversation(user_id)

current_conv_id = st.session_state.current_conv_id


def render_tool_calls(tool_calls: list[dict]):
    """在 UI 上渲染工具调用详情（可折叠），历史消息与本轮流式共用"""
    if not tool_calls:
        return
    with st.expander(f"🔧 本轮调用 {len(tool_calls)} 个工具"):
        for tc in tool_calls:
            st.markdown(f"**{tc.get('name', '')}**")
            if tc.get("args"):
                st.markdown(f"- 参数：`{tc['args']}`")
            if tc.get("result"):
                result_preview = tc["result"][:300]
                if len(tc["result"]) > 300:
                    result_preview += "..."
                st.markdown(f"- 返回：{result_preview}")


# 渲染当前会话消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 如果该轮有工具调用，展示可折叠的详情
        if msg.get("tool_calls"):
            render_tool_calls(msg["tool_calls"])


# ============================================================
# 对话输入与处理
# ============================================================
if prompt := st.chat_input("请输入您的问题，例如：生成我的使用报告 / 扫地机器人怎么保养？"):
    # 记录用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 获取（或创建）agent 实例
    agent = get_agent(middleware_url)

    # 调用 agent，流式接收回答并实时渲染
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        tool_calls = []
        for event in agent.chat_stream(
            user_message=prompt,
            user_id=user_id,
            user_location=user_location,
            conversation_id=current_conv_id,
        ):
            if event["type"] == "text":
                full_text += event["content"]
                placeholder.markdown(full_text)
            else:  # tool_call：本轮工具调用详情
                tool_calls.extend(event["tool_calls"])
                render_tool_calls(event["tool_calls"])
        response = full_text

    # 保存 assistant 消息到显示历史
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "tool_calls": tool_calls,
    })
    # 本轮结束后立即 rerun：侧边栏的代码先于主区域执行，而本轮的结束动作
    # （报告模式自动复位、模式落盘）发生在侧边栏渲染之后。不 rerun 的话，
    # 模式指示器会慢一轮才更新，看起来像"卡在报告模式"。
    st.rerun()


# ============================================================
# 空状态提示
# ============================================================
if not st.session_state.messages:
    st.info("""
    👋 你好！我是扫地机器人智能客服。

    你可以问我：
    - 扫地机器人的使用、保养、故障排查问题
    - 当地天气对扫地机器人使用的影响
    - **生成我的使用报告**（会自动调用工具获取你的使用记录）

    对话结束后点击左侧「🆕 新对话」即可保存为一条历史记录。
    """)

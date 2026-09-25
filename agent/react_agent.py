"""
ReAct Agent 主体

职责：
1. 集成大模型（ChatOpenAI / deepseek）、工具集（build_all_tools）、系统提示词
2. 实现 ReAct 思考循环：思考 -> 调用工具 -> 观察结果 -> 再思考 -> 最终回答
3. 支持用户上下文注入（user_id、user_location）
4. 支持报告模式：检测到 fill_context_for_report 调用后自动切换到报告提示词
5. 与中间件联动：通过 middleware_url 触发上下文注入

使用方式：
    from agent.react_agent import ReactAgent

    agent = ReactAgent(middleware_url="http://127.0.0.1:8000")
    answer = agent.chat(
        user_message="生成我的6月使用报告",
        user_id="1001",
        user_location="深圳",
    )
    print(answer)
"""
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage,
)

from model.factory import get_chat_model
from agent.tools.agents_tools import build_all_tools
from agent.tools.session_store import SessionStore
from utils.prompt_loader import load_system_prompts, load_report_prompts
from utils.logger_handler import logger

# 可选：直接 import 中间件核心函数（不依赖 HTTP 服务）
try:
    from agent.tools import middleware
except ImportError:
    middleware = None


# ============================================================
# 面向用户的兜底文案
# 原则：终端用户只看到可理解的提示，异常原文（含上游 API 报错、栈信息）
# 一律只写日志，不回吐到对话里，避免泄露内部实现与服务端细节。
# ============================================================
MODEL_ERROR_REPLY = "抱歉，模型服务暂时不可用，请稍后重试。若持续失败请联系管理员。"
MAX_ITER_REPLY = "已达到最大工具调用次数，信息仍不足，无法完成回答。请尝试换一种方式提问。"


class ReactAgent:
    """
    ReAct 智能体，支持工具调用与报告模式提示词切换。

    Attributes:
        model: 大语言模型实例（已绑定工具能力）
        main_prompt: 普通客服场景系统提示词
        report_prompt: 报告生成场景系统提示词
        middleware_url: 中间件 HTTP 地址，为 None 时使用本地函数调用
        max_iterations: 单轮对话最大工具调用次数（防止死循环）
        sessions: 会话内存缓存 {conversation_id: {"mode": ..., "history": [...]}}
    """

    def __init__(
        self,
        middleware_url: str = None,
        max_iterations: int = 5,
        session_db_path: str = None,
    ):
        # 惰性获取 chat 模型单例（避免 import 阶段初始化阻塞页面加载）
        self.model = get_chat_model()
        self.middleware_url = middleware_url
        self.max_iterations = max_iterations

        # 加载两套系统提示词
        self.main_prompt = load_system_prompts()
        self.report_prompt = load_report_prompts()

        # 会话状态：以 conversation_id 为 key 维护对话历史和当前模式（内存缓存，落盘在 store）
        self.sessions: dict[str, dict] = {}
        # SQLite 持久化存储：进程重启后可恢复历史会话
        self.store = SessionStore(db_path=session_db_path)

        logger.info(
            f"[ReactAgent] 初始化完成 | 中间件: {middleware_url or '本地函数模式'} "
            f"| 最大迭代: {max_iterations}"
        )

    # ============================================================
    # 会话管理
    # ============================================================

    def _init_session(self, conversation_id: int) -> dict:
        """初始化或获取会话；首次访问时从持久化存储恢复历史"""
        if conversation_id not in self.sessions:
            restored = self.store.get_conversation(conversation_id)
            if restored is None:
                restored = {"mode": "normal", "history": []}
            self.sessions[conversation_id] = {
                "mode": restored["mode"],   # normal | report
                "history": restored["history"],  # LangChain Message 列表
            }
        return self.sessions[conversation_id]

    def _persist_session(self, conversation_id: int, session: dict):
        """将会话（模式 + 完整历史）全量写回 SQLite"""
        self.store.save_conversation(conversation_id, session["mode"], session["history"])

    def get_mode(self, conversation_id: int) -> str:
        """
        获取指定会话的当前模式（"normal" | "report"）。

        会话状态 session["mode"] 是模式的**唯一权威来源**，并随会话一起持久化到
        conversations.mode，进程重启后可从 SQLite 原样恢复。

        此前这里读的是中间件按 user_id 维护的进程内存上下文，会带来两个问题：
        1. 进程重启后内存上下文丢失，历史里的报告会话被套上普通客服提示词；
        2. 同一用户开两个会话时按 user_id 共享状态，会互相干扰。
        中间件不再参与模式决策，仅作镜像同步（见 _switch_to_report_mode）。
        """
        return self._init_session(conversation_id)["mode"]

    def _get_system_prompt(self, conversation_id: int) -> str:
        """根据会话当前模式返回对应系统提示词"""
        if self.get_mode(conversation_id) == "report":
            return self.report_prompt
        return self.main_prompt

    # ============================================================
    # 工具构建与执行
    # ============================================================

    def _build_tools(self, user_id: str, user_location: str):
        """为当前用户构建带状态的工具集"""
        return build_all_tools(
            user_id=user_id,
            user_location=user_location,
            middleware_url=self.middleware_url,
        )

    @staticmethod
    def _execute_tool(tool_call: dict, tools_map: dict) -> str:
        """
        执行单个工具调用，返回观察结果字符串。

        Args:
            tool_call: 模型返回的工具调用字典 {"name":..., "args":..., "id":...}
            tools_map: {工具名: 工具实例}

        Returns:
            工具执行结果的字符串形式
        """
        name = tool_call.get("name")
        args = tool_call.get("args", {})

        tool = tools_map.get(name)
        if tool is None:
            return f"错误：工具 '{name}' 不存在"

        try:
            result = tool.invoke(args)
            return str(result)
        except Exception as e:
            # 异常原文只进日志（含堆栈），交给模型的是可行动的重试指引
            logger.exception(f"[ReactAgent] 工具 {name} 执行失败: {e}")
            return (
                f"工具 {name} 执行失败（内部错误已记录日志）。"
                f"请勿重复调用该工具，可尝试其他工具或直接基于已有信息回答。"
            )

    # ============================================================
    # 核心：ReAct 对话循环
    # ============================================================

    def chat(
        self,
        user_message: str,
        user_id: str = "unknown",
        user_location: str = "未知",
        conversation_id: int = None,
    ) -> str:
        """
        处理一轮用户对话，执行 ReAct 思考循环并返回最终回答。

        Args:
            user_message: 用户输入文本
            user_id: 当前用户 ID（用于工具注入与中间件上下文）
            user_location: 当前用户所在城市
            conversation_id: 当前会话 ID；为 None 时自动新建一个会话

        Returns:
            agent 的最终回答文本
        """
        # 未指定会话时自动创建（独立对话入口可用）
        if conversation_id is None:
            conversation_id = self.store.create_conversation(user_id)

        session = self._init_session(conversation_id)
        session["history"].append(HumanMessage(content=user_message))
        self._persist_session(conversation_id, session)

        # 为本次对话构建工具（用户状态可能变化，每次重新构建）
        tools = self._build_tools(user_id, user_location)
        tools_map = {t.name: t for t in tools}
        model_with_tools = self.model.bind_tools(tools)

        logger.info(
            f"[ReactAgent] 用户 {user_id} 发起对话 | 会话: {conversation_id} "
            f"| 模式: {self.get_mode(conversation_id)} "
            f"| 消息: {user_message[:50]}"
        )

        # ReAct 循环：最多 max_iterations 轮工具调用
        for iteration in range(self.max_iterations):
            # 每轮重新组装消息：系统提示词（可能因模式切换而变化）+ 历史
            system_msg = SystemMessage(content=self._get_system_prompt(conversation_id))
            messages = [system_msg] + session["history"]

            # 调用模型
            try:
                response = model_with_tools.invoke(messages)
            except Exception as e:
                logger.exception(f"[ReactAgent] 模型调用失败: {e}")
                return MODEL_ERROR_REPLY

            session["history"].append(response)
            self._persist_session(conversation_id, session)

            # ---- 情况1：模型发起了工具调用 ----
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    logger.info(
                        f"[ReactAgent] 第 {iteration + 1} 轮 | 调用工具: "
                        f"{tool_name} | 参数: {tool_call.get('args', {})}"
                    )

                    # 检测报告触发工具：切换到报告模式
                    if tool_name == "fill_context_for_report":
                        self._switch_to_report_mode(user_id, conversation_id)

                    # 执行工具，获取观察结果
                    observation = self._execute_tool(tool_call, tools_map)
                    logger.info(
                        f"[ReactAgent] 工具 {tool_name} 返回: {observation[:100]}"
                    )

                    # 将观察结果加入对话历史
                    tool_msg = ToolMessage(
                        content=observation,
                        tool_call_id=tool_call["id"],
                    )
                    session["history"].append(tool_msg)
                    self._persist_session(conversation_id, session)

                # 继续下一轮思考（模型根据观察结果决定继续调用工具或给出答案）
                continue

            # ---- 情况2：模型直接给出最终回答（无工具调用） ----
            answer = response.content or ""
            logger.info(
                f"[ReactAgent] 用户 {user_id} 对话完成 | 迭代 {iteration + 1} 轮 "
                f"| 回答长度: {len(answer)}"
            )
            # 报告已产出：自动退出报告模式，否则用户后续的普通提问
            # （如"怎么保养"）会一直被当成报告场景，这是此前"退不出报告模式"的根因
            if session["mode"] == "report":
                self.finish_report(user_id, conversation_id)
            return answer

        # 达到最大迭代次数仍未得出最终答案
        logger.warning(f"[ReactAgent] 用户 {user_id} 达到最大迭代次数 {self.max_iterations}")
        # 报告流程没跑完也要复位模式：模式现已持久化到 conversations.mode，
        # 若不复位，该会话会永久停在报告提示词下，重启进程也清不掉
        if session["mode"] == "report":
            self.finish_report(user_id, conversation_id)
        return MAX_ITER_REPLY

    # ============================================================
    # 流式对话：ReAct 循环 + 最终回答逐块产出
    # ============================================================

    def chat_stream(
        self,
        user_message: str,
        user_id: str = "unknown",
        user_location: str = "未知",
        conversation_id: int = None,
    ):
        """
        流式版 chat()：执行与 chat() 完全一致的 ReAct 循环，
        但模型的输出文本逐块产出，前端可实时渲染。

        产出事件（dict）：
            {"type": "text", "content": str}              # 模型输出文本增量
            {"type": "tool_call", "tool_calls": [dict]}   # 一轮工具调用（含结果）

        与 chat() 相同的持久化语义：每轮模型输出与工具结果都会
        写入会话历史并落盘，历史会话恢复时内容一致。
        """
        # 未指定会话时自动创建（独立对话入口可用）
        if conversation_id is None:
            conversation_id = self.store.create_conversation(user_id)

        session = self._init_session(conversation_id)
        session["history"].append(HumanMessage(content=user_message))
        self._persist_session(conversation_id, session)

        # 为本次对话构建工具（用户状态可能变化，每次重新构建）
        tools = self._build_tools(user_id, user_location)
        tools_map = {t.name: t for t in tools}
        model_with_tools = self.model.bind_tools(tools)

        logger.info(
            f"[ReactAgent] 用户 {user_id} 发起流式对话 | 会话: {conversation_id} "
            f"| 模式: {self.get_mode(conversation_id)} "
            f"| 消息: {user_message[:50]}"
        )

        # ReAct 循环：最多 max_iterations 轮工具调用
        for iteration in range(self.max_iterations):
            # 每轮重新组装消息：系统提示词（可能因模式切换而变化）+ 历史
            system_msg = SystemMessage(content=self._get_system_prompt(conversation_id))
            messages = [system_msg] + session["history"]

            # 流式消费模型输出：文本增量实时 yield，
            # 同时把所有 chunk 累加成完整消息后再解析工具调用。
            # 注意：langchain 的 chunk.tool_calls 只是部分快照（中间分块 name/args 不完整），
            # 直接取最新会得到 name=''、args={}，因此必须累加 chunk 到 Flow 末尾再取 tool_calls。
            collected_content = []
            accum: AIMessageChunk | None = None
            try:
                for chunk in model_with_tools.stream(messages):
                    if chunk.content:
                        text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                        collected_content.append(text)
                        yield {"type": "text", "content": text}
                    accum = chunk if accum is None else accum + chunk
            except Exception as e:
                # 异常原文只进日志（含堆栈），用户侧只看到可理解的提示
                logger.exception(f"[ReactAgent] 流式模型调用失败: {e}")
                if collected_content:
                    # 已经吐了一部分文本，补一句中断说明，避免用户以为回答完整
                    yield {"type": "text", "content": "\n\n（以上回答因服务异常中断，请重试）"}
                else:
                    yield {"type": "text", "content": MODEL_ERROR_REPLY}
                return

            # 累加完整后解析工具调用（若存在）
            tool_calls = accum.tool_calls if accum is not None else []
            full_content = "".join(collected_content)

            # 重建完整 AIMessage 并写入历史（与 chat() 语义一致）
            response = (
                AIMessage(content=full_content, tool_calls=tool_calls)
                if tool_calls
                else AIMessage(content=full_content)
            )
            session["history"].append(response)
            self._persist_session(conversation_id, session)

            # ---- 情况1：模型发起了工具调用 ----
            if tool_calls:
                executed = []
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "")
                    logger.info(
                        f"[ReactAgent] 第 {iteration + 1} 轮 | 调用工具: "
                        f"{tool_name} | 参数: {tool_call.get('args', {})}"
                    )

                    # 检测报告触发工具：切换到报告模式
                    if tool_name == "fill_context_for_report":
                        self._switch_to_report_mode(user_id, conversation_id)

                    # 执行工具，获取观察结果
                    observation = self._execute_tool(tool_call, tools_map)
                    logger.info(
                        f"[ReactAgent] 工具 {tool_name} 返回: {observation[:100]}"
                    )

                    # 将观察结果加入对话历史
                    tool_msg = ToolMessage(
                        content=observation,
                        tool_call_id=tool_call["id"],
                    )
                    session["history"].append(tool_msg)
                    self._persist_session(conversation_id, session)

                    executed.append({
                        "name": tool_name,
                        "args": tool_call.get("args", {}),
                        "result": observation,
                    })

                # 通知前端展示本轮工具调用，继续下一轮思考
                yield {"type": "tool_call", "tool_calls": executed}
                continue

            # ---- 情况2：模型直接给出最终回答（文本已流式产出完毕） ----
            logger.info(
                f"[ReactAgent] 用户 {user_id} 流式对话完成 | 迭代 {iteration + 1} 轮 "
                f"| 回答长度: {len(full_content)}"
            )
            # 与 chat() 一致：报告已产出即退出报告模式
            if session["mode"] == "report":
                self.finish_report(user_id, conversation_id)
            return

        # 达到最大迭代次数仍未得出最终答案
        logger.warning(f"[ReactAgent] 用户 {user_id} 达到最大迭代次数 {self.max_iterations}")
        # 与 chat() 一致：报告流程没跑完也要复位模式，避免会话永久停在报告提示词下
        if session["mode"] == "report":
            self.finish_report(user_id, conversation_id)
        yield {
            "type": "text",
            "content": MAX_ITER_REPLY,
        }

    # ============================================================
    # 报告模式切换
    # ============================================================

    def _switch_to_report_mode(self, user_id: str, conversation_id: int):
        """
        把指定会话切换到报告模式。

        会话状态 session["mode"] 是模式的**唯一权威来源**：先写会话并落盘，
        提示词切换只依赖它。中间件上下文仅作镜像同步（供其对外 /context/get
        接口与外部集成观察），不参与任何模式决策——进程重启后它丢失也不影响行为。
        """
        session = self._init_session(conversation_id)
        session["mode"] = "report"
        self._persist_session(conversation_id, session)

        # 镜像同步：写失败不影响模式，会话状态已落盘
        if middleware:
            middleware.fill_context(user_id)

        logger.info(
            f"[ReactAgent] 会话 {conversation_id}（用户 {user_id}）已切换到报告模式，"
            f"后续使用报告提示词"
        )

    def finish_report(self, user_id: str = "unknown", conversation_id: int = None):
        """
        退出报告模式：把会话状态恢复为普通模式，并清理中间件中该用户的上下文。

        调用时机：
        - 自动：chat() / chat_stream() 产出一轮的最终回答后，若该会话仍处于
          report 模式，则由内部自动调用——报告已经给出，后续提问应回到普通
          客服模式，否则用户会永久停留在报告提示词下；
        - 手动：前端「退出报告模式」按钮，或调用方在自定义流程结束后自行调用。

        与 _switch_to_report_mode 对称：会话状态是权威源，中间件仅作镜像清理。
        """
        if conversation_id is not None:
            session = self._init_session(conversation_id)
            session["mode"] = "normal"
            self._persist_session(conversation_id, session)
        if middleware:
            middleware.clear_context(user_id)
        logger.info(f"[ReactAgent] 用户 {user_id} 报告完成，已恢复普通模式")


# ============================================================
# 直接运行时的简单测试
# ============================================================
if __name__ == "__main__":
    agent = ReactAgent(middleware_url=None)  # 本地函数模式，无需启动中间件

    # 测试1：普通问答（未指定会话，自动新建）
    print("=" * 50)
    print("测试1：普通问答")
    print("=" * 50)
    conv1 = agent.store.create_conversation("1001")
    print(agent.chat("你好，扫地机器人每天用合适吗？", user_id="1001", user_location="深圳", conversation_id=conv1))

    # 测试2：报告生成（会触发 fill_context_for_report -> 切换报告模式）
    print("\n" + "=" * 50)
    print("测试2：报告生成")
    print("=" * 50)
    conv2 = agent.store.create_conversation("1001")
    print(agent.chat("生成我的使用报告", user_id="1001", user_location="深圳", conversation_id=conv2))

    # 测试3：报告完成后恢复
    agent.finish_report("1001", conversation_id=conv2)

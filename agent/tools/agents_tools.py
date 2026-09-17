import os
import json
import time
import threading
import urllib.request
import urllib.parse
from datetime import datetime

from utils.logger_handler import logger
from langchain_core.tools import tool

from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

# ============================================================
# RAG 懒加载：初始化需连接 Chroma 向量库 + 创建 embedding 模型
# （实测约 28 秒，其中 langchain_chroma 导入 12 秒），若在 import
# 时实例化会导致网页加载极慢。改为首次调用 rag_summarize 工具时
# 才导入 rag 模块并初始化（函数内延迟 import，页面加载完全不受影响）。
# ============================================================
_rag_instance = None
_rag_lock = threading.Lock()


def _get_rag():
    """获取 RAG 服务单例（懒加载 + 延迟导入，线程安全）"""
    global _rag_instance
    if _rag_instance is None:
        with _rag_lock:
            if _rag_instance is None:
                from rag.rag_service import RagSummarizeService
                _rag_instance = RagSummarizeService()
    return _rag_instance


external_data = {}


# ============================================================
# 无状态工具：不依赖当前用户会话，可直接作为模块级变量使用
# ============================================================

# RAG 检索结果缓存：相同问题 10 分钟内直接复用，避免重复检索+重复调模型
_rag_cache: dict[str, tuple[float, str]] = {}
_RAG_CACHE_TTL = 600  # 秒


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    now = time.time()
    hit = _rag_cache.get(query)
    if hit and now - hit[0] < _RAG_CACHE_TTL:
        return hit[1]

    result = _get_rag().rag_summarize(query)
    _rag_cache[query] = (now, result)
    return result


# 天气结果缓存：同城市 30 分钟内直接复用，避免重复请求外部 API
_weather_cache: dict[str, tuple[float, str]] = {}
_WEATHER_CACHE_TTL = 1800  # 秒


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    """调用 wttr.in 免费天气 API 获取实时天气，无需 API Key。"""
    now = time.time()
    cached = _weather_cache.get(city)
    if cached and now - cached[0] < _WEATHER_CACHE_TTL:
        return cached[1]

    try:
        encoded_city = urllib.parse.quote(city)
        url = f"https://wttr.in/{encoded_city}?format=j1&lang=zh"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        current = data["current_condition"][0]
        # 优先取中文描述，兜底取英文
        weather_desc = (
            current.get("lang_zh", [{}])[0].get("value")
            or current["weatherDesc"][0]["value"]
        )
        result = (
            f"城市{city}天气为{weather_desc}，"
            f"气温{current['temp_C']}摄氏度，"
            f"空气湿度{current['humidity']}%，"
            f"{current['winddir16Point']}风{current['windspeedKmph']}公里/小时，"
            f"最近6小时降雨概率极低"
        )
        _weather_cache[city] = (now, result)
        return result
    except Exception as e:
        logger.warning(f"[get_weather] 调用失败: {e}")
        return f"城市{city}天气信息暂时无法获取"


@tool(description="获取当前月份，以纯字符串形式返回，格式YYYY-MM")
def get_current_month() -> str:
    return datetime.now().strftime("%Y-%m")


_external_data_mtime: float = 0.0


def generate_external_data(force: bool = False):
    """
    从 CSV 文件加载外部使用记录数据到 external_data 字典。
    数据结构：{user_id: {month: {"特征":..., "效率":..., "耗材":..., "对比":...}}}

    依据文件 mtime 判断是否需要重新加载：文件未变化时直接复用内存缓存，
    文件被更新后自动重新解析（无需重启进程）。
    """
    global _external_data_mtime

    external_data_path = get_abs_path(agent_conf["external_data_path"])

    if not os.path.exists(external_data_path):
        raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

    mtime = os.path.getmtime(external_data_path)
    if external_data and not force and mtime == _external_data_mtime:
        return

    external_data.clear()
    with open(external_data_path, "r", encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            line = line.strip()
            if not line:
                continue

            arr: list[str] = line.split(",")
            if len(arr) < 6:
                logger.warning(f"[generate_external_data]跳过格式异常行: {line[:50]}")
                continue

            user_id: str = arr[0].replace('"', "")
            feature: str = arr[1].replace('"', "")
            efficiency: str = arr[2].replace('"', "")
            consumables: str = arr[3].replace('"', "")
            comparison: str = arr[4].replace('"', "")
            time: str = arr[5].replace('"', "")

            if user_id not in external_data:
                external_data[user_id] = {}

            external_data[user_id][time] = {
                "特征": feature,
                "效率": efficiency,
                "耗材": consumables,
                "对比": comparison,
            }

    _external_data_mtime = mtime
    logger.info(f"[generate_external_data]已加载外部数据：{len(external_data)} 个用户")


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    try:
        record = external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""

    # 工具描述约定返回纯字符串，这里显式序列化为 JSON 文本，
    # 避免把 dict 直接交给模型（字符串化后是 Python 字面量风格，不易读且引号不规范）
    return json.dumps(record, ensure_ascii=False)


# ============================================================
# 有状态工具：依赖当前登录用户上下文，通过闭包工厂构建
# ============================================================

def build_user_tools(user_id: str, user_location: str, middleware_url: str = None):
    """
    根据当前登录用户构建带状态的工具集。

    在构建 agent 时调用，传入从会话/请求中获取的真实用户信息：
        user_tools = build_user_tools(
            user_id=session["user_id"],
            user_location=session.get("city", "未知"),
            middleware_url="http://your-middleware:8080",  # 可选
        )

    Args:
        user_id: 当前登录用户的 ID
        user_location: 当前用户所在城市
        middleware_url: 报告上下文中间件地址，为 None 时仅记录日志不发请求

    Returns:
        包含 get_user_location、get_user_id、fill_context_for_report 三个工具的列表
    """

    @tool(description="获取用户所在城市的名称，以纯字符串形式返回")
    def get_user_location() -> str:
        return user_location

    @tool(description="获取用户的ID，以纯字符串形式返回")
    def get_user_id() -> str:
        return user_id

    @tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
    def fill_context_for_report():
        try:
            if middleware_url:
                payload = json.dumps({"user_id": user_id}).encode("utf-8")
                req = urllib.request.Request(
                    f"{middleware_url.rstrip('/')}/context/fill",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=3)
            logger.info(f"[fill_context_for_report] 已为用户{user_id}注入报告上下文")
            return "fill_context_for_report已调用"
        except Exception as e:
            logger.warning(f"[fill_context_for_report] 调用失败: {e}")
            return "fill_context_for_report调用失败"

    return [get_user_location, get_user_id, fill_context_for_report]


def build_all_tools(user_id: str = "unknown", user_location: str = "未知", middleware_url: str = None):
    """
    便捷方法：构建完整工具列表（无状态工具 + 用户上下文工具）。

    在 agent 构建处直接调用：
        from agent.tools.agents_tools import build_all_tools
        tools = build_all_tools(
            user_id=session["user_id"],
            user_location=session.get("city", "深圳"),
        )
        agent = create_agent(model, tools, ...)

    Args:
        user_id: 当前登录用户 ID，默认 "unknown"
        user_location: 当前用户所在城市，默认 "未知"
        middleware_url: 报告上下文中间件地址，可选

    Returns:
        完整的 LangChain 工具列表
    """
    stateless_tools = [rag_summarize, get_weather, get_current_month, fetch_external_data]
    user_tools = build_user_tools(user_id, user_location, middleware_url)
    return stateless_tools + user_tools

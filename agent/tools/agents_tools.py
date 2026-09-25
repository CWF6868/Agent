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

# 缓存容量上限：命中 TTL 的条目会被复用，但过期条目不会自动消失。
# 进程长期运行（客服场景常驻）时，不同 query / city 会不断累积，
# 因此在写入时顺带淘汰：先清过期项，仍超上限则丢弃最旧的条目。
_CACHE_MAX_ENTRIES = 128


def _cache_put(cache: dict[str, tuple[float, str]], key: str, value: str, ttl: float) -> None:
    """写入缓存并维持容量上限（先清过期，再按写入时间淘汰最旧）"""
    now = time.time()
    cache[key] = (now, value)
    if len(cache) <= _CACHE_MAX_ENTRIES:
        return

    for k in [k for k, (ts, _) in cache.items() if now - ts >= ttl]:
        cache.pop(k, None)

    while len(cache) > _CACHE_MAX_ENTRIES:
        oldest = min(cache, key=lambda k: cache[k][0])
        cache.pop(oldest, None)


# RAG 检索结果缓存：相同问题 10 分钟内直接复用，避免重复检索+重复调模型
_rag_cache: dict[str, tuple[float, str]] = {}
_RAG_CACHE_TTL = 600  # 秒


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    # 空查询没有任何检索意义：直接返回可行动的提示，
    # 避免把空串送进 embedding 接口与模型（既浪费一次调用，也拿不到有效上下文）
    if not query or not query.strip():
        logger.warning("[rag_summarize] 收到空查询，已跳过检索")
        return "未提供检索内容，请补充具体问题后重试。"

    now = time.time()
    hit = _rag_cache.get(query)
    if hit and now - hit[0] < _RAG_CACHE_TTL:
        return hit[1]

    result = _get_rag().rag_summarize(query)
    _cache_put(_rag_cache, query, result, _RAG_CACHE_TTL)
    return result


# 天气结果缓存：同城市 30 分钟内直接复用，避免重复请求外部 API
_weather_cache: dict[str, tuple[float, str]] = {}
_WEATHER_CACHE_TTL = 1800  # 秒


def _next_6h_rain_desc(data: dict) -> str:
    """
    从 wttr.in j1 响应的 hourly 预报表中取"未来 6 小时"的降雨概率。

    wttr.in 的 weather[0].hourly 提供当天 8 个 3 小时粒度的时段
    （time 为 "0"/"300"/.../"2100"）。这里筛选出**尚未结束**的时段，取最近
    2 个（合计约 6 小时）中的最高降雨概率；当天剩余时段不足 2 个时
    （如 22 点后），用次日的预报顺延补齐。

    判据是「时段结束时刻（起始整点 + 3）晚于当前」，而不是「起始整点 >= 当前」：
    后者会把正在进行的那个时段排除掉，等于把 3 小时后的窗口当成"未来 6 小时"
    上报（例如 04:30 问天气，却报 06:00~12:00 的概率，正好漏掉眼下这段）。
    取 2 个时段而不是 3 个，是为了不把窗口无谓拉长、避免概率被高估。

    取不到任何预报数据时返回空字符串——调用方据此整段省略降雨描述，
    绝不使用固定文案，避免向模型传递与实况无关的结论。
    """
    try:
        days = data.get("weather") or []
        if not days:
            return ""

        now_hour = datetime.now().hour
        hourly = days[0].get("hourly") or []
        # time 为 "0"/"300"/.../"2100"，除以 100 得到时段起始整点
        upcoming = [
            h for h in hourly
            if int(h.get("time", 0)) // 100 + 3 > now_hour
        ]

        if len(upcoming) < 2 and len(days) > 1:
            upcoming += days[1].get("hourly") or []  # 跨天补齐

        slots = upcoming[:2]
        if not slots:
            return ""

        # chanceofrain 偶尔返回空串，用 `or 0` 兜底，避免 int("") 抛异常导致
        # 整段降雨描述被静默丢弃（温度/湿度等有效信息不受影响）
        chances = [int(s.get("chanceofrain") or 0) for s in slots]
        if not chances:
            return ""

        peak = max(chances)
        if peak >= 60:
            level = "较高"
        elif peak >= 30:
            level = "中等"
        else:
            level = "较低"
        return f"未来6小时降雨概率最高{peak}%（{level}）"
    except Exception as e:
        logger.warning(f"[get_weather] 解析降雨概率失败: {e}")
        return ""


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

        # current_condition 同样用 `or [{}]` 兜底，避免整个键缺失时直接 IndexError
        current = (data.get("current_condition") or [{}])[0]
        # 优先取中文描述，兜底取英文。
        # 注意：必须用 `or [{}]` 而不是 `get(key, [{}])`——get 的默认值只在 key 缺失时
        # 生效，key 存在但值为空列表时 [0] 会抛 IndexError，导致整条天气降级为
        # "暂时无法获取"（而 weatherDesc 里其实有可用描述）。
        # weatherDesc 兜底同样处理：两者都取不到时用"未知"，不丢弃温度/湿度等有效信息。
        weather_desc = (
            (current.get("lang_zh") or [{}])[0].get("value")
            or (current.get("weatherDesc") or [{}])[0].get("value")
            or "未知"
        )

        # 各项按需拼接：任一字段缺失只影响它自己那一段，不会让整条天气退化成
        # "暂时无法获取"（原实现用 current['temp_C'] 直取，缺一个键就 KeyError 全盘降级）
        parts = [f"城市{city}天气为{weather_desc}"]
        if current.get("temp_C"):
            parts.append(f"气温{current['temp_C']}摄氏度")
        if current.get("humidity"):
            parts.append(f"空气湿度{current['humidity']}%")
        wind_dir = current.get("winddir16Point")
        wind_speed = current.get("windspeedKmph")
        if wind_dir and wind_speed:
            parts.append(f"{wind_dir}风{wind_speed}公里/小时")
        result = "，".join(parts)

        # 降雨概率为真实预报值；取不到时整段省略，不编造
        rain_desc = _next_6h_rain_desc(data)
        if rain_desc:
            result += f"，{rain_desc}"
        _cache_put(_weather_cache, city, result, _WEATHER_CACHE_TTL)
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


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回。"
                  "若该月无记录，会自动回退到该用户最近一个有数据的月份，并在返回结果中以"
                  "'实际月份'字段说明——此时报告中必须注明实际使用的月份，不得编造当月数据。"
                  "该用户完全无记录时返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    months = external_data.get(user_id)
    if not months:
        logger.warning(f"[fetch_external_data]用户 {user_id} 无任何使用记录")
        return ""

    if month in months:
        # 工具描述约定返回纯字符串，这里显式序列化为 JSON 文本，
        # 避免把 dict 直接交给模型（字符串化后是 Python 字面量风格，不易读且引号不规范）
        return json.dumps(months[month], ensure_ascii=False)

    # 请求月份无数据：回退到语义上最接近的有数据月份，并显式告知模型。
    # 月份键为 "YYYY-MM" 格式，字符串序即时间序，可直接比较。
    available = sorted(months.keys())
    if month < available[0]:
        fallback = available[0]
    elif month > available[-1]:
        fallback = available[-1]
    else:
        # 落在数据区间内的空洞（如某月缺失）：取不晚于请求月份的最近月份
        fallback = max(m for m in available if m <= month)

    logger.warning(
        f"[fetch_external_data]用户 {user_id} 在 {month} 无使用记录，已回退到 {fallback}"
    )
    return json.dumps(
        {
            "实际月份": fallback,
            "用户请求月份": month,
            "数据说明": f"{month} 无使用记录，以下为最近的有数据月份（{fallback}）",
            **months[fallback],
        },
        ensure_ascii=False,
    )


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

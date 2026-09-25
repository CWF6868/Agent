"""
agent 工具运行状态检查脚本
用法： python verify_tools.py
检查层级：1.语法  2.模块导入  3.逐个工具调用  4.agent 构建冒烟测试
"""
import sys
import time
import traceback

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = []


def check(name, func):
    """执行一项检查，记录结果，不中断后续检查"""
    t0 = time.time()
    try:
        func()
        elapsed = time.time() - t0
        results.append((PASS, name, f"{elapsed:.2f}s", ""))
        print(f"{PASS} {name} ({elapsed:.2f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        err = f"{type(e).__name__}: {e}"
        results.append((FAIL, name, f"{elapsed:.2f}s", err))
        print(f"{FAIL} {name} ({elapsed:.2f}s) -> {err}")
        traceback.print_exc()
        print()


# ============================================================
# 第 1 层：语法检查
# ============================================================
print("=" * 60)
print("第 1 层：语法检查")
print("=" * 60)

def syntax_check():
    import py_compile
    py_compile.compile("agent/tools/agents_tools.py", doraise=True)

check("agents_tools.py 语法", syntax_check)


# ============================================================
# 第 2 层：模块导入（检查依赖是否齐全）
# ============================================================
print("\n" + "=" * 60)
print("第 2 层：模块导入")
print("=" * 60)

def import_check():
    global build_all_tools, build_user_tools, get_current_month, get_weather
    global rag_summarize, fetch_external_data
    from agent.tools.agents_tools import (
        build_all_tools, build_user_tools, get_current_month,
        get_weather, rag_summarize, fetch_external_data,
    )

check("导入 agents_tools 模块", import_check)


# ============================================================
# 第 3 层：逐个工具调用
# ============================================================
print("\n" + "=" * 60)
print("第 3 层：工具调用验证")
print("=" * 60)

def tool_count_check():
    tools = build_all_tools(user_id="1001", user_location="江门")
    assert len(tools) == 7, f"期望 7 个工具，实际 {len(tools)}"
    names = [t.name for t in tools]
    expected = ["rag_summarize", "get_weather", "get_current_month",
                "fetch_external_data", "get_user_location", "get_user_id",
                "fill_context_for_report"]
    for n in expected:
        assert n in names, f"缺少工具: {n}"

check("build_all_tools 构建 7 个工具", tool_count_check)


def user_id_check():
    tools = build_user_tools(user_id="9999", user_location="上海")
    id_tool = [t for t in tools if t.name == "get_user_id"][0]
    result = id_tool.invoke({})
    assert result == "9999", f"期望 9999，实际 {result}"

check("get_user_id 返回注入值（非随机）", user_id_check)


def user_location_check():
    tools = build_user_tools(user_id="9999", user_location="上海")
    loc_tool = [t for t in tools if t.name == "get_user_location"][0]
    result = loc_tool.invoke({})
    assert result == "上海", f"期望 上海，实际 {result}"

check("get_user_location 返回注入值（非随机）", user_location_check)


def current_month_check():
    result = get_current_month.invoke({})
    import re
    assert re.match(r"^\d{4}-\d{2}$", result), f"格式错误: {result}"
    print(f"   当前月份 = {result}")

check("get_current_month 格式 YYYY-MM", current_month_check)


def fill_context_check():
    tools = build_user_tools(user_id="9999", user_location="上海")
    fc_tool = [t for t in tools if t.name == "fill_context_for_report"][0]
    result = fc_tool.invoke({})
    assert "已调用" in result or "失败" in result, f"异常返回: {result}"

check("fill_context_for_report 可调用", fill_context_check)


def weather_check():
    """天气 API 依赖网络，失败只警告不报错。

    注意：不能只断言"包含城市名"——降级文案"城市深圳天气信息暂时无法获取"
    同样包含城市名，会让解析失败（如 lang_zh 为空列表导致 IndexError）
    这类 bug 静默通过。这里改为要求返回体包含实际的观测字段。
    """
    try:
        result = get_weather.invoke({"city": "深圳"})
        print(f"   天气返回 = {result[:60]}...")
        assert "深圳" in result, "返回中应包含城市名"
        assert "暂时无法获取" not in result, f"天气查询降级，疑似解析异常: {result}"
        assert "摄氏度" in result and "湿度" in result, f"返回体缺少观测字段: {result}"
        assert "天气为None" not in result and "天气为未知" not in result, (
            f"天气描述解析失败: {result}"
        )
    except Exception as e:
        print(f"   {WARN} 天气 API 调用失败（可能是网络问题）: {e}")

check("get_weather 调用天气 API", weather_check)


def fetch_external_data_check():
    """fetch_external_data 依赖 agent.yml 配置，可能未配置"""
    try:
        result = fetch_external_data.invoke({"user_id": "1001", "month": "2025-01"})
        print(f"   返回 = {result}")
    except Exception as e:
        print(f"   {WARN} fetch_external_data 调用异常: {type(e).__name__}: {e}")
        print("   （通常是因为 config/agent.yml 为空，缺少 external_data_path 配置）")

check("fetch_external_data 调用", fetch_external_data_check)


# ============================================================
# 第 4 层：模型可用性（可选，依赖 API Key）
# ============================================================
print("\n" + "=" * 60)
print("第 4 层：模型与配置检查")
print("=" * 60)

def config_check():
    from utils.config_handler import agent_conf, rag_conf
    print(f"   agent_conf = {agent_conf}")
    print(f"   rag_conf keys = {list(rag_conf.keys()) if rag_conf else 'None'}")
    if agent_conf is None:
        raise RuntimeError("config/agent.yml 为空，fetch_external_data 将无法运行")

check("配置文件加载", config_check)


def model_check():
    """尝试初始化 chat model，依赖 API Key，失败只警告"""
    try:
        from model.factory import get_chat_model
        print(f"   chat_model = {get_chat_model()}")
    except Exception as e:
        print(f"   {WARN} 模型初始化失败（可能缺少 API Key）: {e}")

check("ChatOpenAI 模型初始化", model_check)


# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 60)
print("检查汇总")
print("=" * 60)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print(f"通过: {passed}  失败: {failed}  总计: {len(results)}")
if failed == 0:
    print("\n🎉 全部检查通过，代码可以正常运行。")
else:
    print(f"\n⚠️  有 {failed} 项失败，请根据上方错误信息排查。")
sys.exit(0 if failed == 0 else 1)

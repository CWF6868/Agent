# -*- coding: utf-8 -*-
"""
售后问题批量测试脚本

直接调用 ReactAgent（与 Streamlit app.py 使用同一套代码），对 25 个售后问题
逐一跑 ReAct 循环，自动采集：

  - 最终回答文本（含中间轮的过渡文本，与原实现一致，保证新旧结果可逐字段对比）
  - 每轮工具调用（名称、参数、返回值、是否成功）
  - 首字延迟（从发起请求到产出第一个非空文本块的时间，秒）
  - 总耗时（整轮 ReAct 循环，含全部工具调用）
  - 工具调用总次数与成功次数

输出：
  - eval_results.json   原始结果（逐题完整字段，供人工标注与回归比对）
  - eval_summary.csv    汇总表（含人工标注列；重跑时按 ID 回填，不丢已标注内容）

用法：
    python eval_questions.py                     # 全量 25 题
    python eval_questions.py --ids 21,22,24      # 只跑指定题号（冒烟验证）
    python eval_questions.py --limit 3           # 只跑前 3 题
    python eval_questions.py --session-db data/eval_sessions.db

说明：
  - 模型端点与密钥一律走项目配置（config/rag.yml + config/rag.local.yml 覆盖），
    脚本内不硬编码任何地址或密钥。
  - 默认写入独立的评测会话库（data/eval_sessions.db），不污染生产库
    data/sessions.db——评测会创建大量会话，混入生产库会冲垮历史列表。
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.react_agent import ReactAgent
from utils.logger_handler import logger

# ============================================================
# 测试题库：6 大类 25 题
# ============================================================
QUESTIONS = [
    {"id": 1, "category": "基础使用", "question": "首次使用扫地机器人需要做什么？"},
    {"id": 2, "category": "基础使用", "question": "APP无法连接机器人怎么办？"},
    {"id": 3, "category": "基础使用", "question": "建图不完整、地图错乱怎么解决？"},
    {"id": 4, "category": "基础使用", "question": "如何设置定时清扫？"},
    {"id": 5, "category": "基础使用", "question": "机器人找不到充电座怎么处理？"},
    {"id": 6, "category": "清洁效果", "question": "清扫后地面还有灰尘、碎屑怎么办？"},
    {"id": 7, "category": "清洁效果", "question": "漏扫某个区域怎么解决？"},
    {"id": 8, "category": "清洁效果", "question": "边角位置清扫不干净怎么办？"},
    {"id": 9, "category": "清洁效果", "question": "拖地后地面有水渍、水痕怎么办？"},
    {"id": 10, "category": "清洁效果", "question": "地毯上的灰尘清理不彻底怎么处理？"},
    {"id": 11, "category": "维护耗材", "question": "主刷需要多久更换一次？"},
    {"id": 12, "category": "维护耗材", "question": "HEPA滤网可以清洗重复使用吗？"},
    {"id": 13, "category": "维护耗材", "question": "尘盒多久清理一次？"},
    {"id": 14, "category": "维护耗材", "question": "电池续航越来越短怎么处理？"},
    {"id": 15, "category": "维护耗材", "question": "万向轮卡顿、异响怎么解决？"},
    {"id": 16, "category": "故障报警", "question": "机器人发出滴滴滴报警声是什么原因？"},
    {"id": 17, "category": "故障报警", "question": "传感器异常报警怎么办？"},
    {"id": 18, "category": "故障报警", "question": "滤网堵塞报警如何解决？"},
    {"id": 19, "category": "故障报警", "question": "机器人开机后直接报警关机怎么解决？"},
    {"id": 20, "category": "故障报警", "question": "主刷不旋转怎么办？"},
    {"id": 21, "category": "报告生成", "question": "生成我的使用报告"},
    {"id": 22, "category": "报告生成", "question": "查一下我这个月的机器人使用记录"},
    {"id": 23, "category": "报告生成", "question": "我上个月的扫地机器人使用情况怎么样？"},
    {"id": 24, "category": "天气适配", "question": "今天天气适合用扫地机器人吗？"},
    {"id": 25, "category": "天气适配", "question": "深圳下雨天能用扫地机器人吗？"},
]

DEFAULT_USER_ID = "1001"
DEFAULT_USER_LOCATION = "深圳"
DEFAULT_SESSION_DB = os.path.join("data", "eval_sessions.db")

RESULT_FIELDS = (
    "id", "category", "question", "answer", "tool_calls",
    "tool_call_count", "tool_call_success_count",
    "first_token_latency", "total_time",
)

# CSV 列顺序，与历史产物保持一致；后三列是人工标注位，脚本不生成、只负责保留
CSV_HEADER = [
    "ID", "类别", "问题", "回答摘要", "工具调用", "工具成功数", "工具总数",
    "首字延迟(s)", "总耗时(s)",
    "标注(正确/部分正确/错误)", "失败分类(检索没召回/模型编造/工具选错/无)", "备注",
]
_ANNOTATION_COLS = CSV_HEADER[9:12]

# 工具返回值里出现这些字样即判为失败。工具层降级文案统一带"错误/异常"，
# 与 _execute_tool 的失败返回措辞对齐（见 agent/react_agent.py）。
_FAIL_MARKERS = ("错误", "异常")

# 报告类问题会切换会话模式，跑完需复位，否则模式滞留（幂等兜底）
_REPORT_TRIGGER_TOOL = "fill_context_for_report"


# ============================================================
# 采集
# ============================================================

def _tool_success(result) -> bool:
    """工具调用是否成功：返回值不含失败标记即视为成功"""
    text = str(result or "")
    return not any(marker in text for marker in _FAIL_MARKERS)


def run_single_question(agent: ReactAgent, q: dict, user_id: str, user_location: str) -> dict:
    """
    对单个问题执行一次流式对话，采集全部指标。

    每个问题独立新建会话：报告类问题会把会话切到 report 模式，若与上一题共用
    会话，模式会串到下一题（用错提示词，指标失真）。

    Args:
        agent: ReactAgent 实例
        q: 题目 dict，含 id / category / question
        user_id: 模拟用户 ID（影响取报告、查数据的范围）
        user_location: 模拟所在城市（影响天气工具）

    Returns:
        与 RESULT_FIELDS 对齐的单题结果 dict
    """
    conv_id = agent.store.create_conversation(user_id)
    full_text = ""
    all_tool_calls = []
    first_token_latency = None
    t0 = time.perf_counter()

    try:
        for event in agent.chat_stream(
            user_message=q["question"],
            user_id=user_id,
            user_location=user_location,
            conversation_id=conv_id,
        ):
            etype = event.get("type")

            if etype == "text":
                # 首个非空白文本块到达即首字延迟；空块不计（流式首块常为空串）
                if first_token_latency is None and str(event.get("content", "")).strip():
                    first_token_latency = time.perf_counter() - t0
                full_text += event.get("content", "")

            elif etype == "tool_call":
                for tc in event.get("tool_calls", []):
                    result = tc.get("result", "")
                    all_tool_calls.append({
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {}),
                        "result": result,
                        "success": _tool_success(result),
                    })
    finally:
        total_time = time.perf_counter() - t0
        # 兜底复位：chat_stream 正常收尾时会自行退出报告模式，但中途异常
        # （如模型报错、被中断）会留下 report 状态，这里补一次，幂等。
        if any(tc["name"] == _REPORT_TRIGGER_TOOL for tc in all_tool_calls):
            try:
                agent.finish_report(user_id, conv_id)
            except Exception as e:
                # 清理动作不应掩盖真正的异常，只记日志
                logger.warning(f"[eval] 复位报告模式失败 conv={conv_id}: {e}")

    return {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "answer": full_text,
        "tool_calls": all_tool_calls,
        "tool_call_count": len(all_tool_calls),
        "tool_call_success_count": sum(1 for tc in all_tool_calls if tc["success"]),
        "first_token_latency": (
            round(first_token_latency, 3) if first_token_latency is not None else None
        ),
        "total_time": round(total_time, 3),
    }


# ============================================================
# 输出
# ============================================================

def _summarize(text, width: int = 80) -> str:
    """压成单行摘要，避免回答里的换行破坏 CSV 结构"""
    return " ".join(str(text or "").split())[:width]


def _tool_summary(tool_calls: list) -> str:
    """工具调用摘要，形如 rag_summarize(ok); get_weather(fail)"""
    if not tool_calls:
        return "无工具调用"
    return "; ".join(
        "{}({})".format(tc["name"], "ok" if tc["success"] else "fail") for tc in tool_calls
    )


def load_existing_annotations(csv_path: str) -> dict:
    """
    读回上一轮 CSV 的人工标注列，按 ID 建立映射。

    评测重跑不应抹掉人工标注——那是整份评测里最费人力、且无法再生的产物。
    """
    if not os.path.exists(csv_path):
        return {}

    annotations = {}
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                rid = (row.get("ID") or "").strip()
                if rid:
                    annotations[rid] = tuple(row.get(col, "") or "" for col in _ANNOTATION_COLS)
    except Exception as e:
        print("[WARN] 读取历史 CSV 失败，人工标注将不会保留: {}".format(e))
    return annotations


def write_outputs(results: list, out_dir: str) -> tuple:
    """写 eval_results.json 与 eval_summary.csv，返回两个路径"""
    json_path = os.path.join(out_dir, "eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(out_dir, "eval_summary.csv")
    annotations = load_existing_annotations(csv_path)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for r in results:
            note = annotations.get(str(r["id"]), ("", "", ""))
            writer.writerow([
                r["id"],
                r["category"],
                r["question"],
                _summarize(r["answer"]),
                _tool_summary(r["tool_calls"]),
                r["tool_call_success_count"],
                r["tool_call_count"],
                "" if r["first_token_latency"] is None else r["first_token_latency"],
                r["total_time"],
                *note,
            ])

    return json_path, csv_path


def print_summary(results: list):
    """打印整体指标"""
    total = len(results)
    total_tc = sum(r["tool_call_count"] for r in results)
    total_tc_ok = sum(r["tool_call_success_count"] for r in results)
    latencies = [r["first_token_latency"] for r in results if r["first_token_latency"] is not None]
    total_time = sum(r["total_time"] for r in results)
    errors = [r for r in results if str(r["answer"]).startswith("[TEST_ERROR]")]

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print("  题量      : {}".format(total))
    print("  工具调用  : {} 次，成功 {} 次（{}）".format(
        total_tc, total_tc_ok,
        "{:.1%}".format(total_tc_ok / total_tc) if total_tc else "N/A",
    ))
    if latencies:
        print("  首字延迟  : 平均 {}s | 最快 {}s | 最慢 {}s".format(
            round(sum(latencies) / len(latencies), 3), min(latencies), max(latencies),
        ))
    print("  总耗时    : {}s（平均 {}s/题）".format(
        round(total_time, 1), round(total_time / total, 1) if total else 0,
    ))
    if errors:
        print("  异常题数  : {}".format(len(errors)))
    print("\n下一步：人工填写 eval_summary.csv 的「标注 / 失败分类 / 备注」三列，")
    print("重跑本脚本时这三列会按 ID 自动保留。")


# ============================================================
# 入口
# ============================================================

def select_questions(args) -> list:
    """按 --ids / --limit 筛选题目"""
    questions = list(QUESTIONS)

    if args.ids:
        try:
            wanted = {int(x.strip()) for x in args.ids.replace("，", ",").split(",") if x.strip()}
        except ValueError:
            print("[ERROR] --ids 需为逗号分隔的整数，如 21,22,24")
            sys.exit(1)
        questions = [q for q in questions if q["id"] in wanted]
        missing = wanted - {q["id"] for q in questions}
        if missing:
            print("[WARN] 题库中不存在这些题号，已忽略: {}".format(
                ",".join(str(m) for m in sorted(missing))))
        # 按题号顺序执行，避免输入顺序影响结果顺序
        questions.sort(key=lambda q: q["id"])

    if args.limit > 0:
        questions = questions[:args.limit]

    if not questions:
        print("[ERROR] 筛选后没有可执行的题目，请检查 --ids / --limit")
        sys.exit(1)
    return questions


def main():
    parser = argparse.ArgumentParser(description="扫地机器人智能客服批量评测")
    parser.add_argument("--ids", default="", help="只跑指定题号，逗号分隔，如 21,22,24")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0 表示全部）")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID, help="模拟用户 ID（默认 1001）")
    parser.add_argument("--location", default=DEFAULT_USER_LOCATION, help="模拟所在城市（默认 深圳）")
    parser.add_argument("--session-db", default="", help="评测专用会话库（默认 data/eval_sessions.db）")
    parser.add_argument("--out-dir", default="", help="结果输出目录（默认脚本所在目录；冒烟测试可指向临时目录以免覆盖基线）")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    session_db = args.session_db or os.path.join(root, DEFAULT_SESSION_DB)
    if not os.path.isabs(session_db):
        session_db = os.path.join(root, session_db)

    out_dir = args.out_dir or root
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(root, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    questions = select_questions(args)

    print("=" * 60)
    print("扫地机器人智能客服 —— {} 题批量测试".format(len(questions)))
    print("=" * 60)
    print("\n[1/3] 初始化 Agent（首次需加载 RAG 向量库，约 15~40 秒）...")
    print("      评测会话库: {}（与生产库隔离）".format(session_db))
    print("      预计耗时  : {} 题，约 {:.0f}~{:.0f} 分钟（随模型响应速度浮动）".format(
        len(questions), len(questions) * 30 / 60, len(questions) * 90 / 60))

    agent = ReactAgent(middleware_url=None, session_db_path=session_db)

    # 预热 RAG：首次构造要加载 chromadb 并连上向量库（实测热盘约 15s、冷盘可达 40s）。
    # 这一步必须单独跑掉，否则这份一次性开销会算进第一题的 total_time，
    # Q1 就再也无法与后续题目比较（这与 app 启动时的预热是同一件事）。
    # 单线程顺序执行，写（首次入库）与后续的读不并发，符合 Chroma 的使用约束。
    warm_t0 = time.perf_counter()
    try:
        from rag.rag_service import RagSummarizeService

        RagSummarizeService()
        print("      知识库预热完成，耗时 {:.1f}s（已从后续指标中剥离）".format(
            time.perf_counter() - warm_t0))
    except Exception as e:
        print("      [WARN] 知识库预热失败，首题耗时会偏高（不阻断评测）: {}".format(e))
        logger.warning(f"[eval] 知识库预热失败: {e}")

    # 归档上一次评测遗留的 active 会话，避免库内持续堆积
    archived = agent.store.archive_all_active()
    if archived:
        print("      已归档上一轮遗留会话: {} 个".format(archived))

    print("\n[2/3] 开始测试（用户: {}，城市: {}）\n".format(args.user_id, args.location))

    results = []
    for q in questions:
        print("--- Q{:02d}/{} [{}] {} ---".format(
            q["id"], len(questions), q["category"], q["question"]))

        try:
            result = run_single_question(agent, q, args.user_id, args.location)
            print("  回答: {}...".format(_summarize(result["answer"], 80)))
            print("  工具: {}".format(_tool_summary(result["tool_calls"])))
            latency = result["first_token_latency"]
            print("  首字延迟: {}s | 总耗时: {}s\n".format(
                "N/A" if latency is None else latency, result["total_time"]))
        except Exception as e:
            # 单题失败不中断整轮：记录异常，继续下一题，人工可据 [TEST_ERROR] 定位
            logger.exception(f"[eval] Q{q['id']} 测试异常")
            print("  [ERROR] 测试异常: {}\n".format(repr(e)))
            result = {
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "answer": "[TEST_ERROR] {}".format(repr(e)),
                "tool_calls": [],
                "tool_call_count": 0,
                "tool_call_success_count": 0,
                "first_token_latency": None,
                "total_time": 0.0,
            }

        results.append(result)

    json_path, csv_path = write_outputs(results, out_dir)
    print("[3/3] 原始结果已保存: {}".format(json_path))
    print("      汇总表已保存  : {}".format(csv_path))

    print_summary(results)


if __name__ == "__main__":
    main()

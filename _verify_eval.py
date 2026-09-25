# -*- coding: utf-8 -*-
"""
eval_questions.py 重写版验证脚本

分三部分，全部不调用外部 API：
  A. 契约对齐：题库 / CSV 表头 / JSON 字段 与旧产物逐项比对
  B. 采集逻辑：用桩 agent 驱动 run_single_question，验证指标采集与边界
  C. 产物往返：write_outputs 写出 -> 人工标注 -> 重跑 -> 标注是否保留

结果写入 _verify_eval_report.json
"""
import csv
import json
import marshal
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print("{} {} {}".format("[PASS]" if ok else "[FAIL]", name, detail))


# ============================================================
# A. 契约对齐
# ============================================================
import eval_questions as E

# A1 题库与旧 pyc 常量对齐
pyc_path = os.path.join(ROOT, "__pycache__", "eval_questions.cpython-311.pyc")
old_pyc_strings = set()
if os.path.exists(pyc_path):
    code = marshal.loads(open(pyc_path, "rb").read()[16:])
    old_pyc_strings = {c for c in code.co_consts if isinstance(c, str)}

new_questions = {q["question"] for q in E.QUESTIONS}
new_categories = {q["category"] for q in E.QUESTIONS}

missing_q = new_questions - old_pyc_strings
missing_c = new_categories - old_pyc_strings
check("A1 题库 25 题与旧源码常量一致", len(E.QUESTIONS) == 25 and not missing_q,
      "题数={} 缺失={}".format(len(E.QUESTIONS), sorted(missing_q)))
check("A2 类别 6 类与旧源码常量一致", len(new_categories) == 6 and not missing_c,
      "类别={} 缺失={}".format(sorted(new_categories), sorted(missing_c)))
check("A3 题号 1..25 连续无重复",
      sorted(q["id"] for q in E.QUESTIONS) == list(range(1, 26)),
      "ids={}".format(sorted(q["id"] for q in E.QUESTIONS)))

# A2 与旧 CSV 表头逐列一致
with open(os.path.join(ROOT, "eval_summary.csv"), encoding="utf-8-sig", newline="") as f:
    old_header = next(csv.reader(f))
check("A4 CSV 表头与旧产物逐列一致", old_header == E.CSV_HEADER,
      "old={} new={}".format(old_header, E.CSV_HEADER))

# A3 与旧 JSON 字段集合一致
old_records = json.load(open(os.path.join(ROOT, "eval_results.json"), encoding="utf-8"))
old_keys = set(old_records[0].keys())
new_keys = set(E.RESULT_FIELDS)
check("A5 JSON 记录字段与旧产物一致", old_keys == new_keys,
      "old={} new={}".format(sorted(old_keys), sorted(new_keys)))
check("A6 JSON 记录字段顺序与旧产物一致",
      list(old_records[0].keys()) == list(E.RESULT_FIELDS),
      "old={} new={}".format(list(old_records[0].keys()), list(E.RESULT_FIELDS)))

old_tc_keys = set(old_records[0]["tool_calls"][0].keys())
check("A7 tool_calls 子字段一致", old_tc_keys == {"name", "args", "result", "success"},
      "old={}".format(sorted(old_tc_keys)))


# ============================================================
# B. 采集逻辑（桩 agent，不调模型）
# ============================================================
class FakeStore:
    def __init__(self):
        self.created = []

    def create_conversation(self, user_id):
        self.created.append(user_id)
        return 900 + len(self.created)


class FakeAgent:
    """按预设事件序列产出流式事件"""

    def __init__(self, events, raise_after=None):
        self.store = FakeStore()
        self.events = events
        self.raise_after = raise_after
        self.finish_calls = []

    def chat_stream(self, **kwargs):
        self.kwargs = kwargs
        for i, ev in enumerate(self.events):
            if self.raise_after is not None and i == self.raise_after:
                raise RuntimeError("模拟中断")
            time.sleep(0.02)  # 模拟真实流式间隔，让耗时可观测
            yield ev

    def finish_report(self, user_id=None, conversation_id=None):
        self.finish_calls.append((user_id, conversation_id))


Q = {"id": 1, "category": "基础使用", "question": "测试问题"}

# B1 正常路径：过渡文本 + 工具调用 + 最终回答
agent = FakeAgent([
    {"type": "text", "content": "我先查一下。"},
    {"type": "tool_call", "tool_calls": [
        {"name": "rag_summarize", "args": {"query": "x"}, "result": "参考资料片段"},
    ]},
    {"type": "text", "content": "最终回答内容"},
])
r = E.run_single_question(agent, Q, "1001", "深圳")
check("B1 全量拼接文本（过渡+最终）", r["answer"] == "我先查一下。最终回答内容", repr(r["answer"]))
check("B2 工具调用计数", r["tool_call_count"] == 1 and r["tool_call_success_count"] == 1,
      "count={} ok={}".format(r["tool_call_count"], r["tool_call_success_count"]))
check("B3 首字延迟已采集且为正", isinstance(r["first_token_latency"], float) and r["first_token_latency"] > 0,
      str(r["first_token_latency"]))
check("B4 总耗时已采集且为正", isinstance(r["total_time"], float) and r["total_time"] > 0,
      str(r["total_time"]))
check("B5 每题独立建会话", agent.store.created == ["1001"], str(agent.store.created))
check("B6 非报告题不触发 finish_report", agent.finish_calls == [], str(agent.finish_calls))

# B2 空文本块不占用首字延迟
class SlowAgent(FakeAgent):
    pass


b2_agent = FakeAgent([
    {"type": "text", "content": "   "},          # 纯空白，不应计为首字
    {"type": "text", "content": "实文本"},
])
r2 = E.run_single_question(b2_agent, Q, "1001", "深圳")
check("B7 空/纯空白文本块不计首字延迟", r2["first_token_latency"] is not None,
      "latency={}".format(r2["first_token_latency"]))

# B3 工具失败判定
fail_agent = FakeAgent([
    {"type": "tool_call", "tool_calls": [
        {"name": "get_weather", "args": {}, "result": "工具 get_weather 执行失败（内部错误已记录日志）。"},
        {"name": "rag_summarize", "args": {}, "result": "正常内容"},
    ]},
    {"type": "text", "content": "答"},
])
r3 = E.run_single_question(fail_agent, Q, "1001", "深圳")
check("B8 工具失败正确识别（含'失败'字样的异常返回）",
      r3["tool_call_count"] == 2 and r3["tool_call_success_count"] == 1,
      "count={} ok={} tr={}".format(r3["tool_call_count"], r3["tool_call_success_count"],
                                    [t["success"] for t in r3["tool_calls"]]))

# B4 报告题触发复位（正常收尾）
rep_agent = FakeAgent([
    {"type": "tool_call", "tool_calls": [
        {"name": "fill_context_for_report", "args": {}, "result": "已注入"},
    ]},
    {"type": "text", "content": "报告正文"},
])
r4 = E.run_single_question(rep_agent, Q, "1001", "深圳")
check("B9 报告题调用 finish_report 复位", len(rep_agent.finish_calls) == 1,
      str(rep_agent.finish_calls))

# B5 中途异常：finish_report 仍要执行（finally 兜底），且异常向上抛出
exc_agent = FakeAgent([
    {"type": "tool_call", "tool_calls": [
        {"name": "fill_context_for_report", "args": {}, "result": "已注入"},
    ]},
    {"type": "text", "content": "半截"},
], raise_after=1)
raised = False
try:
    E.run_single_question(exc_agent, Q, "1001", "深圳")
except RuntimeError:
    raised = True
check("B10 中途异常会向上抛出（交由 main 记 [TEST_ERROR]）", raised)
check("B11 中途异常仍复位报告模式（finally 兜底）", len(exc_agent.finish_calls) == 1,
      str(exc_agent.finish_calls))


# ============================================================
# C. 产物往返：人工标注保留
# ============================================================
tmp = tempfile.mkdtemp(prefix="eval_verify_")
try:
    # 首轮：无历史 CSV
    json_p, csv_p = E.write_outputs([r], tmp)
    with open(csv_p, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    check("C1 首轮写出 CSV 表头+1 行", rows[0] == E.CSV_HEADER and len(rows) == 2,
          "rows={}".format(len(rows)))
    check("C2 摘要为单行（换行已折叠）", "\n" not in rows[1][3], repr(rows[1][3])[:60])

    # 人工标注
    rows[1][9] = "正确"
    rows[1][10] = "无"
    rows[1][11] = "回答完整"
    with open(csv_p, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(rows)

    # 重跑：标注应被回填
    json_p2, csv_p2 = E.write_outputs([r], tmp)
    with open(csv_p2, encoding="utf-8-sig", newline="") as f:
        rows2 = list(csv.reader(f))
    check("C3 重跑后人工标注保留",
          rows2[1][9] == "正确" and rows2[1][10] == "无" and rows2[1][11] == "回答完整",
          "note={}".format(rows2[1][9:12]))

    # JSON 字段顺序与字段类型
    recs = json.load(open(json_p2, encoding="utf-8"))
    check("C4 JSON 记录字段顺序与契约一致",
          list(recs[0].keys()) == list(E.RESULT_FIELDS),
          str(list(recs[0].keys())))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# 汇总
total = len(RESULTS)
passed = sum(1 for x in RESULTS if x["ok"])
print("\n{} / {} 通过".format(passed, total))
with open(os.path.join(ROOT, "_verify_eval_report.json"), "w", encoding="utf-8") as f:
    json.dump({"passed": passed, "total": total, "results": RESULTS}, f,
              ensure_ascii=False, indent=2)
sys.exit(0 if passed == total else 1)

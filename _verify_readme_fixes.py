"""
验证「README 失实纠正 + README 指引触发的代码缺陷」这批改动。

覆盖：
  A. 知识文件后缀过滤（大小写 / 多余点号 / 目录 / 无点号文件名）
  B. get_file_md5_hex 失败返回 None，且不会再被当成"已入库"
  C. 首次同步：入库 + 记录 MD5
  D. 幂等：再跑一次不重复写入
  E. 文件内容变更：旧分片被真正删除（检索不到旧内容）
  F. 文件被删除：分片与 MD5 记录一并清除
  G. 只删 chroma_db（md5.txt 仍在）→ 自动重新入库，不再静默失效
  H. 只删 md5.txt → 全量重建且不产生重复分片
  I. 三个 yaml 加载器对空文件返回 {}（不再返回 None 把报错推到远处）
  J. get_weather 字段缺失时不再整条降级为"暂时无法获取"
  K. 文件仍在但内容不可入库（清空 / 全空白 / 编码损坏）时，旧分片同样被清除
     —— 与 E 同属「文件改动后旧向量不清理」，是同一症状的另一条触发路径

不触碰任何生产数据：向量库/数据目录/MD5 记录全部指向 .workbuddy/_tmp_readme_fix/，
embedding 用确定性假实现（不发网络请求），会调用模型的部分全部打桩。
"""
import json
import math
import os
import re
import shutil
import sys
import zlib

ROOT = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(ROOT, ".workbuddy", "_tmp_readme_fix")
sys.path.insert(0, ROOT)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"item": name, "ok": bool(ok), "detail": str(detail)})
    print("[{}] {}  {}".format("PASS" if ok else "FAIL", name, detail))
    return ok


def info(name, detail=""):
    RESULTS.append({"item": name, "ok": True, "detail": str(detail), "info": True})
    print("[INFO] {}  {}".format(name, detail))


# ============================================================
# 确定性假 embedding：词袋哈希成定长向量。
# 同一个词的文档向量相近，因此可以用关键词判断"某段内容还在不在库里"。
# ============================================================
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    DIM = 64

    @staticmethod
    def _vec(text: str):
        v = [0.0] * FakeEmbeddings.DIM
        for token in re.findall(r"\w+", (text or "").lower()):
            v[zlib.crc32(token.encode("utf-8")) % FakeEmbeddings.DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def main():
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    data_dir = os.path.join(TMP, "data")
    os.makedirs(data_dir)

    # ------------------------------------------------------------
    # 把向量库相关配置指向临时目录（就地修改共享的字典对象）
    # ------------------------------------------------------------
    import utils.config_handler as ch

    ch.chroma_conf.update({
        "collection_name": "agent_readme_fix_test",
        "persist_directory": ".workbuddy/_tmp_readme_fix/chroma_db",
        "data_path": ".workbuddy/_tmp_readme_fix/data",
        "md5_hex_store": ".workbuddy/_tmp_readme_fix/md5.txt",
        "k": 3,
        "chunk_size": 200,
        "chunk_overlap": 20,
        "separators": ["\n\n", "。", ".", "?", "？", "!", " ", ""],
    })
    md5_store = os.path.join(TMP, "md5.txt")

    from utils.file_handler import listdir_with_allowed_type, get_file_md5_hex
    from utils.prompt_loader import get_abs_path

    # ============================================================
    # A. 后缀过滤
    # ============================================================
    open(os.path.join(data_dir, "a.TXT"), "w", encoding="utf-8").write(
        "甲文件 关于吸力档位的说明 关键词天狼星。"
    )
    open(os.path.join(data_dir, "b.txt"), "w", encoding="utf-8").write(
        "乙文件 关于滤网更换周期的说明 关键词猎户座。"
    )
    open(os.path.join(data_dir, "c_pdf"), "w", encoding="utf-8").write("无点号文件名，不该被当成 pdf")
    open(os.path.join(data_dir, "d.md"), "w", encoding="utf-8").write("不在允许类型内")
    os.makedirs(os.path.join(data_dir, "e.txt"), exist_ok=True)  # 目录，名字像 txt

    got = sorted(os.path.basename(p) for p in listdir_with_allowed_type(data_dir, ("txt", "pdf")))
    check("A1 无点号配置 [\"txt\",\"pdf\"] 仍可用，且后缀大小写不敏感", got == ["a.TXT", "b.txt"], got)

    got2 = sorted(os.path.basename(p) for p in listdir_with_allowed_type(data_dir, (".TXT", ".PDF")))
    check("A2 带点 + 大写配置 [\".TXT\",\".PDF\"] 等价", got2 == ["a.TXT", "b.txt"], got2)

    check("A3 无点号文件名 c_pdf 被正确排除", "c_pdf" not in got, got)
    check("A4 未允许类型 d.md 被排除", "d.md" not in got, got)
    check("A5 以 .txt 结尾的目录 e.txt 被排除", "e.txt" not in got, got)

    # ============================================================
    # B. md5 失败返回 None
    # ============================================================
    check(
        "B1 文件不存在时 get_file_md5_hex 返回 None",
        get_file_md5_hex(os.path.join(data_dir, "不存在的文件.txt")) is None,
    )
    check("B2 正常文件返回 32 位 hex", (get_file_md5_hex(os.path.join(data_dir, "b.txt")) or "").__len__() == 32)

    # ============================================================
    # 打桩 embedding，构造向量库服务
    # ============================================================
    import rag.vector_store as vsmod
    vsmod.get_embed_model = lambda: FakeEmbeddings()

    vs = vsmod.VectorStoreService()
    # 环境自检：确认真的连的是临时库，而不是生产库
    check(
        "S0 已隔离到临时向量库（集合名独立 + 落盘目录在临时区）",
        vs.vector_store._collection_name == "agent_readme_fix_test"
        and os.path.isdir(os.path.join(TMP, "chroma_db")),
        "collection={} persist={}".format(
            vs.vector_store._collection_name, os.path.join(TMP, "chroma_db")
        ),
    )

    def count():
        return vs._count_vectors()

    def read_md5_lines():
        if not os.path.exists(md5_store):
            return []
        return [l.strip() for l in open(md5_store, encoding="utf-8") if l.strip()]

    def retrieve_texts(query, k=5):
        return [d.page_content for d in vs.vector_store.similarity_search(query, k=k)]

    # ============================================================
    # C. 首次同步
    # ============================================================
    # 追加第三个文件用于"内容变更"场景
    solo = os.path.join(data_dir, "solo.txt")
    open(solo, "w", encoding="utf-8").write("丙文件 关于尘盒清理的说明 关键词蓝色独角兽。")

    s1 = vs.load_document()
    c1 = count()
    check("C1 首次同步：3 个文件全部入库", s1["added"] == 3, s1)
    check("C2 向量库有分片", c1 > 0, "count={}".format(c1))
    check("C3 MD5 记录行数 == 入库文件数", len(read_md5_lines()) == 3, read_md5_lines())
    check("C4 无点号 / 未允许文件未入库", c1 == 3, "count={}（预期 3，每文件 1 个分片）".format(c1))

    # ============================================================
    # D. 幂等
    # ============================================================
    s2 = vs.load_document()
    check("D1 重复执行全部跳过（幂等）", s2["unchanged"] == 3, s2)
    check("D2 重复执行不新增分片", count() == c1, "count {} -> {}".format(c1, count()))
    check("D3 MD5 记录未被重复追加", len(read_md5_lines()) == 3, read_md5_lines())

    # ============================================================
    # E. 文件内容变更 → 旧分片必须被删除
    # ============================================================
    before_hit = retrieve_texts("蓝色独角兽")
    check("E0 变更前能检索到旧内容", any("蓝色独角兽" in t for t in before_hit), "命中 {} 条".format(len(before_hit)))

    open(solo, "w", encoding="utf-8").write("丙文件 关于尘盒清理的说明 关键词红色凤凰。")
    s3 = vs.load_document()
    check("E1 变更被识别为 update（不是新增）", s3["updated"] == 1 and s3["added"] == 0, s3)
    check("E2 分片总数不变（旧分片被替换，不是追加）", count() == c1, "count {} -> {}".format(c1, count()))

    after_hit = retrieve_texts("蓝色独角兽")
    check(
        "E3 旧内容已检索不到（证明旧分片真的被删了）",
        not any("蓝色独角兽" in t for t in after_hit),
        "命中 {} 条".format(len(after_hit)),
    )
    new_hit = retrieve_texts("红色凤凰")
    check("E4 新内容可被检索到", any("红色凤凰" in t for t in new_hit), "命中 {} 条".format(len(new_hit)))

    # MD5 记录被重写：不应同时留下新旧两个摘要
    lines = read_md5_lines()
    check("E5 MD5 记录整体重写，不含旧版本摘要", len(lines) == 3, lines)

    # ============================================================
    # F. 文件被删除
    # ============================================================
    os.remove(os.path.join(data_dir, "a.TXT"))
    s4 = vs.load_document()
    check("F1 删除的文件被识别", s4["removed"] == 1, s4)
    check("F2 其分片已清除", count() == c1 - 1, "count {} -> {}".format(c1, count()))
    check("F3 MD5 记录同步减少到 2 行", len(read_md5_lines()) == 2, read_md5_lines())
    check(
        "F4 已删文件的内容检索不到",
        not any("天狼星" in t for t in retrieve_texts("天狼星")),
        retrieve_texts("天狼星"),
    )

    # ============================================================
    # G. 只删 chroma_db（保留 md5.txt）→ 自动重新入库
    #    等价做法：直接清空集合（磁盘索引没了，md5.txt 还在）。
    #    为避免在进程内删目录后复用 Chroma 句柄带来的干扰，这里清空集合模拟。
    # ============================================================
    md5_lines_before = read_md5_lines()
    all_ids = vs.vector_store.get(include=[])["ids"]
    vs.vector_store.delete(ids=all_ids)
    check("G0 已模拟出「库为空但 md5.txt 有记录」的状态", count() == 0 and len(md5_lines_before) == 2,
          "count={} md5记录={}".format(count(), md5_lines_before))

    s5 = vs.load_document()
    check("G1 检测到不同步后自动重新入库（不再静默失效）", s5["added"] == 2 and s5["unchanged"] == 0, s5)
    check("G2 向量库已恢复", count() == c1 - 1, "count={}".format(count()))

    # ============================================================
    # H. 只删 md5.txt → 全量重建且不产生重复分片
    # ============================================================
    before_h = count()
    os.remove(md5_store)
    s6 = vs.load_document()
    # 注意：库内原本已有这些文件的分片，因此走的是"替换"分支（记在 updated 而非 added）——
    # 这正是想要的行为：先删旧分片再入库，所以不会翻倍。
    check("H1 记录缺失触发逐文件重新入库", s6["updated"] == 2 and s6["unchanged"] == 0, s6)
    check("H2 未产生重复分片（先删旧分片）", count() == before_h, "count {} -> {}".format(before_h, count()))
    check("H3 MD5 记录被重建", len(read_md5_lines()) == 2, read_md5_lines())

    # ============================================================
    # I. yaml 空文件兜底
    # ============================================================
    empty = os.path.join(TMP, "empty.yml")
    open(empty, "w", encoding="utf-8").close()
    for fn_name in ("load_chroma_config", "load_prompts_config", "load_agent_config"):
        fn = getattr(ch, fn_name)
        try:
            got_conf = fn(empty)
            check("I-{} 空文件返回空 dict".format(fn_name), got_conf == {}, repr(got_conf))
        except Exception as e:
            check("I-{} 空文件返回空 dict".format(fn_name), False, "抛异常: {}: {}".format(type(e).__name__, e))

    # ============================================================
    # J. get_weather 字段缺失不再整条降级
    # ============================================================
    import urllib.request
    import agent.tools.agents_tools as at

    class FakeResp:
        def __init__(self, payload):
            self._b = json.dumps(payload).encode("utf-8")

        def read(self):
            return self._b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    real_urlopen = urllib.request.urlopen
    try:
        # J1: 缺 humidity / 风向 / 风速，但描述与温度可用
        at._weather_cache.clear()
        urllib.request.urlopen = lambda *a, **k: FakeResp({
            "current_condition": [{"lang_zh": [{"value": "多云"}], "temp_C": "26"}],
            "weather": [],
        })
        r1 = at.get_weather.invoke("深圳")
        check(
            "J1 部分字段缺失时不再整体降级为“暂时无法获取”",
            "暂时无法获取" not in r1 and "气温26摄氏度" in r1,
            r1,
        )
        check("J2 缺失字段被单独略去，保留可用信息", "湿度" not in r1 and "多云" in r1, r1)

        # J3: 全字段齐备时输出格式与改动前完全一致（回归）
        at._weather_cache.clear()
        urllib.request.urlopen = lambda *a, **k: FakeResp({
            "current_condition": [{
                "lang_zh": [{"value": "多云"}],
                "temp_C": "26", "humidity": "78",
                "winddir16Point": "东南", "windspeedKmph": "5",
            }],
            "weather": [],
        })
        r2 = at.get_weather.invoke("广州")
        check(
            "J3 字段齐备时输出格式未变（回归）",
            r2 == "城市广州天气为多云，气温26摄氏度，空气湿度78%，东南风5公里/小时",
            r2,
        )

        # J4: current_condition 整体缺失也不崩
        at._weather_cache.clear()
        urllib.request.urlopen = lambda *a, **k: FakeResp({"weather": []})
        r3 = at.get_weather.invoke("上海")
        check("J4 current_condition 缺失时仍返回可用文本", "天气为" in r3 and "暂时无法获取" not in r3, r3)
    finally:
        urllib.request.urlopen = real_urlopen
        at._weather_cache.clear()

    # ============================================================
    # K. 文件仍在磁盘上、但内容已不可入库 → 旧分片同样必须清除
    #
    # 与 E 是同一个症状（"文件改动后旧向量不清理"）的不同触发路径：
    # 文件被清空 / 只剩空白 / 变成非法编码导致解析器抛错。此前这三条分支只
    # 跳过、不删旧分片，库里会继续提供上一版内容。
    # ============================================================
    def assert_cleared(fname, new_content, label, keyword):
        """写入新内容后同步，断言该文件的旧分片已从库里清除。"""
        path = os.path.join(data_dir, fname)
        open(path, "w", encoding="utf-8").write(
            "{} 关于某项维护的说明 关键词{}。".format(label, keyword)
        )
        vs.load_document()
        check(
            "K-{}-子 变更前旧内容可检索".format(fname),
            any(keyword in t for t in retrieve_texts(keyword)),
            "",
        )

        if isinstance(new_content, bytes):
            with open(path, "wb") as f:
                f.write(new_content)
        else:
            open(path, "w", encoding="utf-8").write(new_content)

        st = vs.load_document()
        still = [t for t in retrieve_texts(keyword) if keyword in t]
        check(
            "K-{} {} 旧分片已清除且不再被检索到".format(fname, label),
            not still and st["cleared"] >= 1,
            "stats={} 仍命中的旧内容条数={}".format(st, len(still)),
        )

    assert_cleared("清空.txt", "", "被清空(0字节)", "翡翠海豚")
    assert_cleared("空白.txt", "   \n\n  \t \n", "只剩空白字符", "琥珀松鼠")
    assert_cleared("乱码.txt", b"\xff\xfe\x00\x01\x80\x81", "非法编码(解析抛错)", "靛蓝羚羊")

    # 清空类文件不应污染 MD5 记录（不记录 = 下次启动重新评估，修好后自动入库）
    lines_k = read_md5_lines()
    check(
        "K4 不可入库的文件不写入 MD5 记录",
        len(lines_k) == 2,
        lines_k,
    )

    # ============================================================
    # 汇总
    # ============================================================
    failed = [r for r in RESULTS if not r["ok"]]
    report = {
        "total": len([r for r in RESULTS if not r.get("info")]),
        "passed": len([r for r in RESULTS if r["ok"] and not r.get("info")]),
        "failed": len(failed),
        "results": RESULTS,
        "tmp_dir": TMP,
    }
    out = os.path.join(ROOT, "_verify_readme_fixes_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n合计 {}/{} 通过，失败 {} 项".format(report["passed"], report["total"], report["failed"]))
    print("报告：{}".format(out))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

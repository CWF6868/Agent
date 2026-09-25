import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import get_embed_model
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=get_embed_model(),
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

        
    # 获取检索器,用于"根据查询找最相关的文档片段"
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def _add_documents_in_batches(self, documents: list, batch_size: int = 10):
        """把文档按 batch_size 分批写入向量库，规避 embedding 服务单次条数上限。

        阿里云 text-embedding 接口单次上限为 20 条，超过会返回
        400 InvalidParameter: batch size is invalid，因此这里留出余量取 10。
        """
        for i in range(0, len(documents), batch_size):
            self.vector_store.add_documents(documents[i : i + batch_size])

    # ==========================================================
    # MD5 记录：只是"该文件已同步"的加速标记，不是权威数据源
    # ==========================================================
    def _md5_store_path(self) -> str:
        return get_abs_path(chroma_conf["md5_hex_store"])

    def _read_md5_store(self) -> set:
        path = self._md5_store_path()
        if not os.path.exists(path):
            return set()
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    def _write_md5_store(self, md5_set: set) -> None:
        """按本次同步的实际结果**整体重写**记录文件。

        不追加而是重写，是为了让记录与向量库内容严格对应：追加会让旧版本文件的
        MD5 永久留在文件里，一旦用户把文件改回旧版本，就会被误判为"已入库"而跳过
        （库里其实只剩新版本的分片）。
        """
        with open(self._md5_store_path(), "w", encoding="utf-8") as f:
            for md5_hex in sorted(md5_set):
                f.write(md5_hex + "\n")

    # ==========================================================
    # 向量库按"文件(source)"维度操作：同一文件的分片可整批替换/删除
    # ==========================================================
    def _sources_in_store(self) -> set:
        """返回向量库中已存在的全部 source（文件绝对路径）。"""
        try:
            metas = self.vector_store.get(include=["metadatas"])["metadatas"]
        except Exception as e:
            logger.error(f"[加载知识库]读取向量库元数据失败：{e}", exc_info=True)
            return set()
        return {m["source"] for m in metas if m and m.get("source")}

    def _delete_by_source(self, source: str) -> int:
        """删除某个文件在向量库中的全部分片，返回实际删除的条数。

        依据是入库时写入的 metadata.source（PyPDFLoader / TextLoader 都会把它设为
        文件路径），因此这里做的是"按文件"删除，而不是按 MD5 或按内容猜。
        """
        try:
            ids = self.vector_store.get(where={"source": source}, include=[])["ids"]
        except Exception as e:
            logger.error(f"[加载知识库]查询待删除分片失败（{source}）：{e}", exc_info=True)
            return 0

        if not ids:
            return 0

        self.vector_store.delete(ids=ids)
        return len(ids)

    def _count_vectors(self) -> int:
        try:
            return self.vector_store._collection.count()
        except Exception:
            # 兜底：不使用私有属性时退化为按 id 计数
            return len(self.vector_store.get(include=[])["ids"])

    def _load_file_documents(self, read_path: str) -> list:
        """按扩展名分派到对应加载器（大小写不敏感）。"""
        ext = os.path.splitext(read_path)[1].lower()
        if ext == ".txt":
            return txt_loader(read_path)
        if ext == ".pdf":
            return pdf_loader(read_path)
        logger.warning(f"[加载知识库]{read_path}后缀 {ext} 没有对应加载器，跳过")
        return []

    def load_document(self) -> dict:
        """把 data/ 下的知识文件增量同步进向量库（新增 / 修改 / 删除三种情况都处理）。

        同步规则（每个文件以其绝对路径作为 source 定位）：
        - **新增**：切分入库并记录其 MD5
        - **修改**：文件内容变了 → MD5 变了 → 先删掉该文件的**旧分片**，再入库新分片。
          此前只做追加，导致同一文件的新旧两版内容同时留在库里、都能被检索到
          （回答会引用已经过期的知识），且向量库随每次编辑持续膨胀。
        - **删除**：磁盘上已不存在的文件，其分片与 MD5 记录一并清除。
          此前只追加不清理，已删除的内容仍可被检索。

        MD5 记录与向量库必须一致，否则会出现「6 个文件全被跳过 → 检索永远返回空 →
        RAG 静默失效」。这里不靠人工保证，而是每次同步都校验、都能自愈：
        - 只删了 chroma_db/ 而留下 md5.txt → 检测到库内无对应分片 → 重新入库
        - 只删了 md5.txt → 记录为空 → 全量重新入库（先删旧分片，不会产生重复）
        - 文件被改回旧版本 → 该 MD5 已不在重写后的记录里 → 重新入库

        :return: 统计字典 {"added","updated","unchanged","removed","skipped"}
        """
        allowed_files: list = list(
            listdir_with_allowed_type(
                get_abs_path(chroma_conf["data_path"]),
                tuple(chroma_conf["allow_knowledge_file_type"]),
            )
        )
        allowed_set = set(allowed_files)

        recorded_md5 = self._read_md5_store()
        sources_in_store = self._sources_in_store()

        if recorded_md5 and not sources_in_store:
            logger.warning(
                "[加载知识库]md5.txt 有记录但向量库中没有任何分片（两者不同步），"
                "本次按全量重新入库处理"
            )
        elif not recorded_md5 and sources_in_store:
            logger.warning(
                "[加载知识库]md5.txt 不存在或为空，但向量库中已有分片，"
                "本次逐文件重新入库（会先删掉各自旧分片，不会产生重复）"
            )

        stats = {"added": 0, "updated": 0, "unchanged": 0, "removed": 0, "skipped": 0}
        kept_md5: set = set()

        for path in allowed_files:
            md5_hex = get_file_md5_hex(path)

            if md5_hex is None:
                # 文件读不到（被独占锁定 / 权限不足 / 列目录后被移走）。
                # 既不能记录 MD5，也不能动库里已有的分片——静观其变，下次启动重试。
                logger.error(f"[加载知识库]{path}无法计算 MD5，本次跳过（下次启动会重试）")
                stats["skipped"] += 1
                continue

            if md5_hex in recorded_md5 and path in sources_in_store:
                logger.info(f"[加载知识库]{path} 内容已存在知识库内，跳过")
                kept_md5.add(md5_hex)
                stats["unchanged"] += 1
                continue

            is_update = path in sources_in_store

            try:
                # 用对应加载器把文件读成 Document 列表
                documents: list[Document] = self._load_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    stats["skipped"] += 1
                    continue

                # 把 Document 列表切分成小块
                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    stats["skipped"] += 1
                    continue

                # 先清旧分片再写新分片。顺序是有意的：
                # 若写入中途失败，MD5 不会被记录，下次启动会再进本分支，
                # 把这次残留的半成品分片一起清掉，自行收敛（反过来先写后删则会留下重复分片）。
                removed = self._delete_by_source(path)
                sources_in_store.discard(path)

                self._add_documents_in_batches(split_document)
                sources_in_store.add(path)

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                kept_md5.add(md5_hex)

                if is_update:
                    stats["updated"] += 1
                    # 措辞不写"内容已变化"：走到这里也可能是 md5.txt 缺失导致的重跑，
                    # 内容其实没变。只陈述"替换了旧分片"这一事实。
                    logger.info(
                        f"[加载知识库]{path} 替换旧分片 {removed} 条并重新入库"
                        f"（{len(split_document)} 个新分片）"
                    )
                else:
                    stats["added"] += 1
                    logger.info(
                        f"[加载知识库]{path} 内容加载成功（{len(split_document)} 个分片）"
                    )
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                stats["skipped"] += 1
                continue

        # 清理磁盘上已删除的文件：分片与 MD5 记录一起移除。
        # 注意只清理"不在磁盘清单里"的 source——清单内但本次没处理成功的
        # （例如上面 md5 计算失败）必须保留，否则会误删有效数据。
        for stale_source in sorted(sources_in_store - allowed_set):
            removed = self._delete_by_source(stale_source)
            stats["removed"] += 1
            logger.info(f"[加载知识库]文件已不存在，移除其分片 {removed} 条：{stale_source}")

        self._write_md5_store(kept_md5)

        logger.info(
            "[加载知识库]同步完成：新增 {added} / 更新 {updated} / 未变 {unchanged} / "
            "移除 {removed} / 跳过 {skipped}，向量库当前共 {total} 条".format(
                total=self._count_vectors(), **stats
            )
        )
        return stats


if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-"*20)


"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import get_chat_model
from utils.logger_handler import logger


class RagSummarizeService:
    def __init__(self):
         # 实例化向量库服务(内部会连接 Chroma)
        self.vector_store = VectorStoreService()
         # 增量入库：RAG 服务初始化时把 data/ 下新增的知识文件补进向量库。
        # 入库按文件 MD5 去重，已入库的文件直接跳过，因此重复调用是幂等的。
        # 有了这一步，新增或修改知识文件后无需再手动执行 python rag/vector_store.py。
        self._ensure_knowledge_loaded()
         # 获取检索器(封装了 top-k 相似度检索)
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
         # 把提示词文本转成 PromptTemplate 对象
        # 模板中通常包含 {input} 和 {context} 两个占位符
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = get_chat_model()
        self.chain = self._init_chain()

    def _ensure_knowledge_loaded(self):
        """同步知识库增量。失败只记录日志，不阻断 RAG 服务启动。"""
        try:
            self.vector_store.load_document()
        except Exception as e:
            logger.error(
                f"[RagSummarizeService] 知识库增量入库失败，本次跳过: {e}", exc_info=True
            )

    def _init_chain(self):
        return self.prompt_template | self.model | StrOutputParser()
    
    # 检索方法:根据查询词返回相关文档列表
    def retriever_docs(self, query: str) -> list[Document]:
        try:
            return self.retriever.invoke(query)
        except Exception as e:
            # Chroma 的 hnsw 索引句柄在向量库被外部重建后会失效，典型报错：
            # Error creating hnsw segment reader: Nothing found on disk。
            # 此时磁盘上的索引本身是完好的，重建客户端即可恢复；
            # 仍失败则向上抛出，由调用方降级处理。
            logger.warning(f"[RagSummarizeService] 检索失败，重建向量库客户端后重试：{e}")
            self.vector_store = VectorStoreService()
            self.retriever = self.vector_store.get_retriever()
            return self.retriever.invoke(query)
    
     # RAG 核心方法:检索 + 拼上下文 + 调用模型生成回答
    def rag_summarize(self, query: str) -> str:

        context_docs = self.retriever_docs(query)

        context = ""
        counter = 0
        for doc in context_docs:
             # 每篇文档加上编号和元数据,方便模型区分来源
            # 格式:【参考资料1】: 参考资料：xxx | 参考元数据：{...}
            counter += 1                    
            
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"#metadata记录这份数据的**来源、属性、上下文信息**

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))

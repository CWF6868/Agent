import os
from abc import ABC, abstractmethod

from utils.config_handler import rag_conf

# 定义抽象基类 BaseModelFactory,继承自 ABC
# 作用:作为所有"模型工厂"的统一父类,规范子类必须实现 generator 方法
class BaseModelFactory(ABC):
     # 使用 @abstractmethod 声明抽象方法,子类必须实现,否则无法实例化
    @abstractmethod
    def generator(self):
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self):
        # 延迟导入：langchain_openai 模块导入约 7.5 秒，放函数内避免拖慢页面加载
        from langchain_openai import ChatOpenAI

        # API 端点优先级：配置文件 api_base > 环境变量 OPENAI_API_BASE > OpenAI 官方默认
        api_base = rag_conf.get("api_base") or os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")

        kwargs = {
            "model": rag_conf["chat_model_name"],
            "timeout": 120,       # 单次请求超时 120 秒（ReAct 工具调用耗时较长）
            "max_retries": 2,     # 超时/网络错误时自动重试 2 次
        }
        if api_base:
            kwargs["base_url"] = api_base

        return ChatOpenAI(**kwargs)


class EmbeddingsFactory(BaseModelFactory):
    def generator(self):
        from langchain_community.embeddings import DashScopeEmbeddings
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


# ============================================================
# 惰性单例：chat_model / embed_model 初始化耗时数秒到数十秒，
# 改为首次调用时才创建，避免 import 阶段阻塞页面加载。
# ============================================================
_chat_model = None
_embed_model = None


def get_chat_model():
    """获取 chat 模型单例（惰性创建，约 3-4 秒，仅首次）"""
    global _chat_model
    if _chat_model is None:
        _chat_model = ChatModelFactory().generator()
    return _chat_model


def get_embed_model():
    """获取 embedding 模型单例（惰性创建，仅首次）"""
    global _embed_model
    if _embed_model is None:
        _embed_model = EmbeddingsFactory().generator()
    return _embed_model

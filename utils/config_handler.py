"""
统一配置加载入口。

职责：
1. 在项目最早被导入的位置读取根目录下的 .env（即 .env.example 的复制目标）
2. 加载 config/ 下的各 YAML 配置，并以模块级常量对外暴露

为什么 .env 要在这里读：本模块被 model.factory / agent / rag / utils 等各处
导入，是导入链上最早的位置之一，能保证任何模型客户端（ChatOpenAI、
DashScopeEmbeddings）在构造前 OPENAI_API_KEY、DASHSCOPE_API_KEY 已经就位。
load_dotenv 默认不覆盖已存在的环境变量，因此用系统环境变量或容器注入的
部署方式不受影响，.env 只起补充作用。
"""
import os

import yaml

from utils.path_tool import get_abs_path

try:
    from dotenv import load_dotenv
except ImportError:  # 未安装 python-dotenv：降级为只使用系统环境变量
    load_dotenv = None

_ENV_PATH = get_abs_path(".env")

if load_dotenv is not None and os.path.exists(_ENV_PATH):
    # override=False：已存在的系统环境变量优先，.env 只做补充
    load_dotenv(_ENV_PATH, override=False)


def load_rag_config(encoding: str="utf-8"):
    """
    加载模型相关配置：以 config/rag.yml 为基线，若存在 config/rag.local.yml
    则用其覆盖同名字段（只覆盖、不删除基线里的其他键）。

    config/rag.local.yml 被 .gitignore 忽略，用于存放含真实端点的本地配置，
    README 的配置步骤即按此约定撰写。此前只读 rag.yml，导致按文档创建的
    rag.local.yml 从未生效，端点只能落到空值、模型名也只能用仓库里的占位值。
    """
    def _read(path: str) -> dict:
        with open(path, "r", encoding=encoding) as f:
            return yaml.load(f, Loader=yaml.FullLoader) or {}

    conf = _read(get_abs_path("config/rag.yml"))

    local_path = get_abs_path("config/rag.local.yml")
    if os.path.exists(local_path):
        conf.update(_read(local_path))

    return conf


def load_chroma_config(config_path: str=get_abs_path("config/chroma.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()


if __name__ == '__main__':
    print(rag_conf["chat_model_name"])

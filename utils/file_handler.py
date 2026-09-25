import os
import hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def get_file_md5_hex(filepath: str):     # 获取文件的md5的十六进制字符串；计算失败返回 None

    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return None

    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return None

    md5_obj = hashlib.md5()

    chunk_size = 4096       # 4KB分片，避免文件过大爆内存
    try:
        with open(filepath, "rb") as f:     # 必须二进制读取
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)

            """
            chunk = f.read(chunk_size)
            while chunk:
                
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None


def _normalize_allowed_types(allowed_types) -> set:
    """把后缀配置统一成 {".txt", ".pdf"} 这种带点、全小写的形式。

    这样 ["txt","pdf"]、[".txt",".pdf"]、[".TXT"] 三种写法等价，
    调用方不必关心配置里有没有点、大小写怎么写。
    """
    normalized = set()
    for item in allowed_types or ():
        item = (item or "").strip().lower()
        if not item:
            continue
        normalized.add(item if item.startswith(".") else "." + item)
    return normalized


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):        # 返回文件夹内的文件列表（允许的文件后缀）
    """列出文件夹内符合后缀要求的**文件**，返回绝对路径元组。

    后缀比较统一按「取扩展名 + 转小写」进行，修复了此前用 str.endswith 判断的两个问题：
    - 大小写敏感：`a.TXT` 在 Windows 上被静默漏掉（用户以为已入库，其实没有）
    - 不校验点号：无点的名字如 `c_pdf` 会因为 endswith("pdf") 被误判为知识文件
    """
    normalized = _normalize_allowed_types(allowed_types)
    files = []

    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return tuple()

    for f in os.listdir(path):
        full_path = os.path.join(path, f)
        if os.path.splitext(f)[1].lower() in normalized and os.path.isfile(full_path):
            files.append(full_path)

    return tuple(files)


def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    return PyPDFLoader(filepath, passwd).load()


def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()

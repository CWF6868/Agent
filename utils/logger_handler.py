import logging
import logging.handlers
from utils.path_tool import get_abs_path
import os
from datetime import datetime
"""

"""
#日志保存的根目录
LOG_ROOT = get_abs_path("logs")

#确保日志的目录存在
os.makedirs(LOG_ROOT ,exist_ok= True)

#日志配置格式
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)#%(filename)s	产生日志的文件名,%(lineno)d	产生日志的行号,%(message)s	日志正文内容

def get_logger(
        name:str = "agent",
        console_level:int = logging.DEBUG,
        file_level :int = logging.DEBUG,
        log_file = None #日志文件路径，不传则自动生成
)-> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    
    # 避免重复添加Handler
    if logger.handlers:
        return logger
    
#Handler（处理器）决定"日志最终发到哪里、以什么级别过滤、按什么格式写
# 控制台Handler
    console_handler = logging.StreamHandler()#默认输出到 sys.stderr
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(console_handler)

    # 文件Handler
    # 用 RotatingFileHandler 而非 FileHandler：日志文件名按天生成，但单日量并不小
    # （实测单日约 100KB，接入更多工具调用后会更高），不轮转的话文件会无限增长。
    # 单文件 5MB、保留 5 个滚动备份，足以覆盖历史排查需求。
    if not log_file:        # 日志文件的存放路径
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(file_handler)

    return logger


# 快捷获取日志器
logger = get_logger()

if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")

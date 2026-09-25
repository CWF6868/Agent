"""
报告上下文中间件

职责：
1. 接收 agent 中 fill_context_for_report 工具的调用，为指定用户注入报告生成上下文
2. 提供上下文查询、清理接口（供外部集成观察当前状态）
3. 对外暴露 HTTP 接口（POST /context/fill、GET /context/get、POST /context/clear）

⚠️ 模式归属说明：
    模式的**权威来源是会话状态**（ReactAgent.sessions 的 "mode" →
    持久化为 conversations.mode，按 conversation_id 隔离）。本模块的
    _context_store 只是按 user_id 维护的进程内存镜像，agent 写入它仅为兼容
    对外接口，**不参与提示词切换决策**；进程重启后它丢失也不影响 agent 行为。

接口：
  POST /context/fill   body: {"user_id": "1001"}  -> 注入报告上下文
  GET  /context/get?user_id=1001                  -> 查询当前上下文
  POST /context/clear  body: {"user_id": "1001"}  -> 清理上下文，恢复普通模式

使用方式：
  python middleware.py              # 启动中间件服务（默认 127.0.0.1:8000）
  或在代码中：from middleware import fill_context, get_context, clear_context
"""
import json
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from utils.logger_handler import logger

# ============================================================
# 线程安全的用户上下文存储
# ============================================================
_context_store: dict[str, dict] = {}
_store_lock = threading.Lock()

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


# ============================================================
# 核心操作函数（可直接 import 调用，无需启动 HTTP 服务）
# ============================================================

def fill_context(user_id: str, extra: dict = None) -> dict:
    """
    为指定用户注入报告生成上下文。
    注入后该用户的模式变为 report，agent 应切换到报告提示词。

    Args:
        user_id: 用户 ID
        extra: 额外上下文字段，会合并到上下文中

    Returns:
        注入后的完整上下文字典
    """
    with _store_lock:
        ctx = {
            "user_id": user_id,
            "mode": "report",
            "prompt_type": "report",
            "filled_at": datetime.now().isoformat(),
        }
        if extra:
            ctx.update(extra)
        _context_store[user_id] = ctx
    logger.info(f"[middleware] 已为用户 {user_id} 注入报告上下文，模式切换为 report")
    return ctx


def get_context(user_id: str) -> dict:
    """
    获取用户当前上下文（进程内存镜像，非模式权威来源）。
    若用户无上下文，返回默认普通模式。

    Returns:
        {"mode": "normal"|"report", "prompt_type": "main"|"report", ...}
    """
    with _store_lock:
        ctx = _context_store.get(user_id)
        if ctx:
            return dict(ctx)
    return {"mode": "normal", "prompt_type": "main"}


def clear_context(user_id: str) -> bool:
    """
    清理用户上下文，恢复普通模式。
    报告生成完成后应调用此方法。

    Returns:
        True 表示有上下文被清理，False 表示用户本就无上下文
    """
    with _store_lock:
        if user_id in _context_store:
            del _context_store[user_id]
            logger.info(f"[middleware] 已清理用户 {user_id} 的上下文，恢复普通模式")
            return True
    return False


def is_report_mode(user_id: str) -> bool:
    """快捷判断镜像上下文中该用户是否为报告模式（仅反映最近一次 fill/clear，非权威来源）"""
    return get_context(user_id).get("mode") == "report"


# ============================================================
# HTTP 服务实现
# ============================================================

class MiddlewareHandler(BaseHTTPRequestHandler):
    """处理中间件 HTTP 请求"""

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_POST(self):
        if self.path == "/context/fill":
            data = self._read_json()
            user_id = data.get("user_id")
            if not user_id:
                self._send_json({"error": "user_id is required"}, 400)
                return
            ctx = fill_context(user_id, data)
            self._send_json({"status": "ok", "context": ctx})

        elif self.path == "/context/clear":
            data = self._read_json()
            user_id = data.get("user_id")
            if not user_id:
                self._send_json({"error": "user_id is required"}, 400)
                return
            clear_context(user_id)
            self._send_json({"status": "ok"})

        else:
            self._send_json({"error": "not found"}, 404)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/context/get":
            params = parse_qs(parsed.query)
            user_id = params.get("user_id", [None])[0]
            if not user_id:
                self._send_json({"error": "user_id is required"}, 400)
                return
            ctx = get_context(user_id)
            self._send_json({"context": ctx})

        elif parsed.path == "/health":
            self._send_json({"status": "running"})

        else:
            self._send_json({"error": "not found"}, 404)

    def log_message(self, format, *args):
        # 用项目日志器代替默认 stderr 输出
        logger.debug(f"[middleware] HTTP {args[0]}")


def start_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    """启动中间件 HTTP 服务（阻塞运行）"""
    server = HTTPServer((host, port), MiddlewareHandler)
    logger.info(f"[middleware] 中间件服务启动: http://{host}:{port}")
    logger.info(f"[middleware] 可用接口: POST /context/fill, GET /context/get, POST /context/clear, GET /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("[middleware] 收到中断信号，服务停止")
        server.server_close()


# ============================================================
# 直接运行时启动服务
# ============================================================
if __name__ == "__main__":
    start_server()

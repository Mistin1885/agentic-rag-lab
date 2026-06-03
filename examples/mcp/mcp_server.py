#!/usr/bin/env python
"""統一 MCP Server — 供 03 與 04 notebook 共用。

Tools:
  add(a, b)                            整數加法（03 示範用）
  greet(name)                          中文問候（03 示範用）
  search_meetings(pattern, max_results)  搜尋 CRM 會議紀錄（04 示範用）
  read_meeting(filename)               讀取完整會議紀錄（04 示範用）

Resources:
  info://server-description            本 server 的工具清單說明
  crm://customers                      CRM 客戶名稱清單（動態萃取自檔名）
  crm://tags                           CRM 標籤 taxonomy

Prompts:
  analyze_customer(customer_name)      分析指定客戶的 prompt 模板
"""
import sys
from pathlib import Path

# utils/ 在 examples/ 下，從 examples/mcp/ 往上一層找
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP

from utils.tools import FileTools

MCP_HOST = "127.0.0.1"
MCP_PORT = 3100

mcp = FastMCP("crm-demo-server", host=MCP_HOST, port=MCP_PORT)
_crm = FileTools(Path(__file__).parent.parent.parent / "data" / "crm_notes")


# ── Demo Tools (03) ───────────────────────────────────────────

@mcp.tool()
def add(a: int, b: int) -> int:
    """將兩個整數相加並回傳結果。"""
    return a + b


@mcp.tool()
def greet(name: str) -> str:
    """以繁體中文問候指定的人。"""
    return f"你好，{name}！歡迎使用 MCP。"


# ── CRM Tools (04) ────────────────────────────────────────────

@mcp.tool()
def search_meetings(pattern: str, max_results: int = 10) -> list[str]:
    """在 CRM 會議紀錄中搜尋關鍵詞，回傳 'file:lineno: text' 清單。"""
    return _crm.grep(pattern, max_results)


@mcp.tool()
def read_meeting(filename: str) -> str:
    """讀取指定的會議紀錄檔案，回傳完整內容。"""
    return _crm.read_file(filename)


# ── Resources ─────────────────────────────────────────────────

@mcp.resource("info://server-description")
def server_description() -> str:
    """本 server 的工具、resource、prompt 清單說明。"""
    return (
        "crm-demo-server 提供以下原語：\n"
        "\n"
        "Tools:\n"
        "  add(a, b)                         整數加法\n"
        "  greet(name)                       中文問候\n"
        "  search_meetings(pattern, max)     搜尋 CRM 會議紀錄\n"
        "  read_meeting(filename)            讀取完整會議紀錄\n"
        "\n"
        "Resources:\n"
        "  info://server-description         本說明文字\n"
        "  crm://customers                   CRM 客戶清單\n"
        "  crm://tags                        標籤 taxonomy\n"
        "\n"
        "Prompts:\n"
        "  analyze_customer(customer_name)   客戶分析模板"
    )


@mcp.resource("crm://customers")
def list_customers() -> str:
    """CRM 所有客戶名稱清單，每行一個（動態從檔名萃取）。"""
    customers: set[str] = set()
    for fname in _crm.list_files():
        # 格式：meeting_NNN_客戶名稱_YYYY-MM-DD.md
        parts = Path(fname).stem.split("_")
        if len(parts) >= 3:
            customers.add(parts[2])
    return "\n".join(sorted(customers))


@mcp.resource("crm://tags")
def tag_taxonomy() -> str:
    """CRM 標籤分類 taxonomy（風險等級、狀態、文件類型）。"""
    return (
        "風險等級: 高, 中, 低\n"
        "嚴重度: 高, 中, 低\n"
        "機率: 高, 中, 低\n"
        "狀態: 待處理, 進行中, 已完成, 已取消\n"
        "文件類型: 會議紀錄, 技術規格, 報價單\n"
        "部署方案: 公有雲, 私有雲, 混合雲"
    )


# ── Prompts ───────────────────────────────────────────────────

@mcp.prompt()
def analyze_customer(customer_name: str) -> str:
    """產生分析指定客戶所有會議紀錄的 prompt 模板。"""
    return (
        f"請分析客戶「{customer_name}」的所有會議紀錄，回答以下問題：\n"
        "1. 主要需求與痛點為何？\n"
        "2. 目前有哪些高風險項目？\n"
        "3. 尚未完成的行動項目（TODO）有哪些？\n"
        "請引用來源檔名。"
    )


if __name__ == "__main__":
    print(f"MCP server 啟動於 http://{MCP_HOST}:{MCP_PORT}/sse")
    mcp.run(transport="sse")

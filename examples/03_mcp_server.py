#!/usr/bin/env python
"""最基礎的 MCP Server 範例 — demo-server。

工具：
  add(a, b)   整數加法
  greet(name) 中文問候

資源：
  info://server-description  本 server 的簡介文字
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """將兩個整數相加並回傳結果。"""
    return a + b


@mcp.tool()
def greet(name: str) -> str:
    """以繁體中文問候指定的人。"""
    return f"你好，{name}！歡迎使用 MCP。"


@mcp.resource("info://server-description")
def server_description() -> str:
    """這個 demo-server 的簡介與可用工具清單。"""
    return (
        "demo-server 是一個示範用的 MCP Server。\n"
        "可用工具：\n"
        "  - add(a, b)   : 整數加法\n"
        "  - greet(name) : 中文問候\n"
        "可用資源：\n"
        "  - info://server-description : 本說明文字"
    )


if __name__ == "__main__":
    mcp.run()  # 預設以 stdio 模式啟動

"""FastAPI 应用入口"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.chat import router as chat_router
from api.kb import router as kb_router
from rag.indexer import index_sample_docs

# ------------------------------------------------------------------
# 应用初始化
# ------------------------------------------------------------------
app = FastAPI(
    title="RAGFlow Agent",
    description="基于 LangChain 的 RAG 知识问答 + Agent 工具调用系统",
    version="1.0.0",
)

# CORS（开发时允许前端跨域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router)
app.include_router(kb_router)

import asyncio

@app.on_event("startup")
async def startup():
    """应用启动时后台异步索引示例文档，不阻塞服务"""
    print("[startup] 服务已就绪，后台正在加载 RAG 知识库...", flush=True)

    async def _build_index():
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, index_sample_docs)
            print("[startup] 知识库加载完成", flush=True)
        except Exception as e:
            print(f"[startup] 警告: 知识库初始化失败({e})，RAG 检索暂不可用，服务继续运行", flush=True)

    asyncio.create_task(_build_index())

@app.get("/")
async def root():
    return {
        "service": "RAGFlow Agent",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "POST /api/chat        — 多轮对话（含工具调用）",
            "POST /api/chat/stream — 流式对话（SSE）",
            "POST /api/kb/import   — 导入知识库文档",
            "GET  /api/kb/search   — 直接检索知识库",
            "GET  /api/kb/status   — 查看知识库状态",
            "DELETE /api/kb        — 清空知识库",
        ],
    }

"""API 路由 — 对话接口"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional

from agent.core import get_agent, AgentResponse

router = APIRouter(prefix="/api", tags=["chat"])

class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    session_id: Optional[str] = Field(None, description="会话 ID，不传则自动创建")

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_calls: List[dict] = []
    sources: List[str] = []

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    多轮对话接口
    Agent 自动决定是否需要调用工具（计算器 / 天气 / RAG 检索）
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    agent = get_agent()
    result: AgentResponse = agent.run(req.message, req.session_id or "")
    return ChatResponse(
        session_id=result.session_id,
        reply=result.reply,
        tool_calls=result.tool_calls,
        sources=result.sources,
    )

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    流式对话接口（SSE）
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    agent = get_agent()

    async def event_stream():
        import asyncio
        # 先执行完整 Agent 逻辑（工具调用部分），再逐字输出
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: agent.run(req.message, req.session_id or ""))
        for char in result.reply:
            import json
            yield f"data: {json.dumps({'char': char})}\n\n"
        yield f"data: {__import__('json').dumps({'done': True, 'session_id': result.session_id, 'tool_calls': result.tool_calls, 'sources': result.sources})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """获取指定会话历史（调试用）"""
    from agent.memory import SessionStore
    session = SessionStore.get(session_id)
    if not session.messages:
        raise HTTPException(status_code=404, detail="会话不存在或无消息")
    return {
        "session_id": session_id,
        "messages": [
            {"type": m.type, "content": m.content}
            for m in session.messages
        ],
    }

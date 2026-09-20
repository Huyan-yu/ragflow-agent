"""
会话记忆管理 — 维护每个 session 的多轮对话历史
"""
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import List, Dict
import uuid
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage

@dataclass
class Session:
    session_id: str
    messages: List[BaseMessage] = field(default_factory=list)
    max_turns: int = 10   # 保留最近 N 轮对话

    def add_user(self, text: str):
        self.messages.append(HumanMessage(content=text))
        self._trim()

    def add_ai(self, text: str):
        self.messages.append(AIMessage(content=text))
        self._trim()

    def add_tool_result(self, tool_name: str, result: str, call_id: str = ""):
        self.messages.append(ToolMessage(
            content=result,
            name=tool_name,
            tool_call_id=call_id or str(uuid.uuid4()),
        ))

    def _trim(self):
        """保留最近 max_turns * 2 条消息（user + ai）"""
        limit = self.max_turns * 2
        if len(self.messages) > limit:
            self.messages = self.messages[-limit:]

    def get_context(self) -> List[BaseMessage]:
        """返回当前对话上下文（供 LLM 使用）"""
        return self.messages

# ------------------------------------------------------------------
# 全局会话池
# ------------------------------------------------------------------
class SessionStore:
    _sessions: Dict[str, Session] = {}

    @classmethod
    def get(cls, session_id: str = "") -> Session:
        if not session_id or session_id not in cls._sessions:
            sid = session_id or str(uuid.uuid4())
            cls._sessions[sid] = Session(session_id=sid)
            session_id = sid
        return cls._sessions[session_id]

    @classmethod
    def reset(cls, session_id: str):
        cls._sessions.pop(session_id, None)

    @classmethod
    def list_active(cls) -> List[str]:
        return list(cls._sessions.keys())

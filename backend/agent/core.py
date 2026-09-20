"""
Agent 核心逻辑 — 多轮对话 + 工具调用 + RAG 增强
"""
from dataclasses import dataclass, field
from typing import List, Optional
import uuid

from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage,
)

from agent.llm import get_llm
from agent.tools import get_all_tools
from agent.memory import SessionStore

SYSTEM_PROMPT = """\
你是一个智能 AI 助手，可以调用工具来帮助用户。

你的工具列表：
1. calculate — 计算数学表达式
2. get_weather — 查询城市天气
3. search_knowledge_base — 检索本地知识库

规则：
- 当用户问题可以通过工具解决时，主动调用对应工具。
- 工具返回结果后，用自然语言总结答案，不要原样输出工具结果。
- 如果知识库有相关内容，在回答末尾标注「参考来源：xxx」。
- 回答简洁、准确，语气友好。
"""

@dataclass
class AgentResponse:
    session_id: str
    reply: str
    tool_calls: List[dict] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)

class RAGAgent:
    """
    核心 Agent：
    1. 接收用户消息
    2. 构建带工具列表的 LLM 请求
    3. 循环执行：LLM 决定是否调用工具 -> 执行工具 -> 将结果回传 LLM
    4. 最终生成回答
    """

    def __init__(self, max_tool_rounds: int = 3):
        self.max_tool_rounds = max_tool_rounds
        self.tools = get_all_tools()
        self.tool_map = {t.name: t for t in self.tools}

    def run(self, user_message: str, session_id: str = "") -> AgentResponse:
        session = SessionStore.get(session_id)
        session.add_user(user_message)

        # 构建 messages 列表
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + session.get_context()
        llm = get_llm()

        # 将工具绑定到 LLM（LangChain 0.2+ 方式）
        llm_with_tools = llm.bind_tools(self.tools)

        tool_calls_log: List[dict] = []
        sources: List[str] = []
        reply = ""

        for _ in range(self.max_tool_rounds + 1):
            response = llm_with_tools.invoke(messages)

            # 检查是否有工具调用
            if hasattr(response, "tool_calls") and response.tool_calls:
                messages.append(response)

                for tc in response.tool_calls:
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("args", {})
                    call_id = tc.get("id", str(uuid.uuid4()))

                    print(f"[TOOL] 调用 {tool_name}({tool_args})", flush=True)

                    tool_fn = self.tool_map.get(tool_name)
                    if tool_fn is None:
                        result_text = f"未知工具：{tool_name}"
                    else:
                        try:
                            result_text = tool_fn.invoke(tool_args)
                        except Exception as e:
                            result_text = f"工具 {tool_name} 执行失败：{str(e)}"

                    tool_calls_log.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result_text[:200],  # 截断存储
                    })

                    # 记录来源（RAG 工具）
                    if tool_name == "search_knowledge_base":
                        for line in result_text.split("\n---\n"):
                            if "来源:" in line:
                                src = line.split("来源:")[-1].split("\n")[0].strip()
                                if src and src not in sources:
                                    sources.append(src)

                    session.add_tool_result(tool_name, result_text, call_id)
                    messages.append(ToolMessage(
                        content=result_text,
                        name=tool_name,
                        tool_call_id=call_id,
                    ))
                continue  # 让 LLM 继续处理工具结果

            # 没有工具调用，最终回答
            reply = response.content
            session.add_ai(reply)
            break

        return AgentResponse(
            session_id=session.session_id,
            reply=reply,
            tool_calls=tool_calls_log,
            sources=sources,
        )

    def stream_run(self, user_message: str, session_id: str = ""):
        """
        简化版流式：先执行完整 Agent 逻辑，再逐字 yield
        真实项目中可改用 LLM 的 stream 模式
        """
        result = self.run(user_message, session_id)
        for char in result.reply:
            yield char
        yield "\n"  # 结束标记

# 全局单例
_agent: Optional[RAGAgent] = None

def get_agent() -> RAGAgent:
    global _agent
    if _agent is None:
        _agent = RAGAgent()
    return _agent

"""
LLM 封装层 — 兼容 OpenAI API 格式的大模型调用
支持：OpenAI / DeepSeek / Moonshot / 其他兼容 API
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def get_llm(temperature: float | None = None) -> ChatOpenAI:
    """获取 LLM 实例，每次调用都重新读取 .env"""
    load_dotenv()
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.7")),
    )

"""
RAG 检索器封装 — 提供独立的检索接口（供 API 直接调用）
"""
from typing import List
from langchain_core.documents import Document
from rag.indexer import get_retriever

def retrieve(query: str, top_k: int = 4) -> List[Document]:
    """
    直接检索知识库，返回最相关的 top_k 个文档片段
    """
    retriever = get_retriever()
    if retriever is None:
        return []
    return retriever.invoke(query)

def retrieve_with_sources(query: str, top_k: int = 4) -> List[dict]:
    """
    检索并格式化输出，包含来源信息
    """
    docs = retrieve(query, top_k)
    return [
        {
            "content": d.page_content,
            "source": d.metadata.get("source", "unknown"),
            "type": d.metadata.get("type", "unknown"),
        }
        for d in docs
    ]

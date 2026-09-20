"""
文本分块器 — 将长文档切分为适合向量化的片段
"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter
import json

# 分块参数
CHUNK_SIZE = 512    # 目标每块 token 数（约字符数）
CHUNK_OVERLAP = 64  # 相邻块重叠字符数

def chunk_markdown(content: str, source_name: str = "document") -> List[Document]:
    """将 Markdown 文本分块"""
    splitter = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = splitter.split_text(content)
    return [
        Document(
            page_content=d,
            metadata={"source": source_name, "type": "markdown"},
        )
        for d in docs
    ]

def chunk_json(content: str, source_name: str = "document") -> List[Document]:
    """将 JSON 文件分块（每个顶层对象作为一个 chunk）"""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return chunk_markdown(content, source_name)

    docs: List[Document] = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            text = json.dumps(item, ensure_ascii=False, indent=2)
            docs.append(Document(
                page_content=text,
                metadata={"source": f"{source_name}[{i}]", "type": "json"},
            ))
    else:
        text = json.dumps(data, ensure_ascii=False, indent=2)
        docs.extend(chunk_markdown(text, source_name))
    return docs

def chunk_documents(file_content: str, filename: str) -> List[Document]:
    """根据文件类型选择分块策略"""
    if filename.endswith(".json"):
        return chunk_json(file_content, filename)
    return chunk_markdown(file_content, filename)

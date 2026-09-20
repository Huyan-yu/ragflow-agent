"""
FAISS 向量索引构建与管理
使用纯 Python 本地 Embedding，无需外部 API
"""
import os
import glob
import hashlib
from typing import List, Optional, Dict
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from rag.chunker import chunk_documents


class LocalHashEmbeddings(Embeddings):
    """纯 Python 实现的轻量本地 Embedding，兼容 LangChain Embeddings 接口"""
    dim: int = 256

    def _hash_embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = text.lower().split()
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[h] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_embed(text)


_local_emb: Optional[LocalHashEmbeddings] = None


def _get_local_embeddings() -> LocalHashEmbeddings:
    global _local_emb
    if _local_emb is None:
        _local_emb = LocalHashEmbeddings()
    return _local_emb


_vector_store: Optional[FAISS] = None
_index_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "faiss_index")


def build_index_from_files(file_paths: List[str]) -> int:
    """从文件列表构建（或更新）FAISS 索引，返回索引中的文档数量"""
    global _vector_store
    emb = _get_local_embeddings()
    all_docs: List[Document] = []
    for fp in file_paths:
        p = Path(fp)
        if not p.exists():
            continue
        content = p.read_text(encoding="utf-8", errors="ignore")
        all_docs.extend(chunk_documents(content, p.name))
    if not all_docs:
        return 0
    if _vector_store is None:
        _vector_store = FAISS.from_documents(all_docs, emb)
    else:
        _vector_store.add_documents(all_docs)
    _save_index()
    return len(all_docs)


def _save_index():
    """将 FAISS 索引持久化到磁盘"""
    if _vector_store is None:
        return
    os.makedirs(_index_dir, exist_ok=True)
    _vector_store.save_local(_index_dir)


def _load_index() -> bool:
    """启动时尝试加载已保存的索引"""
    global _vector_store
    index_file = os.path.join(_index_dir, "index.faiss")
    if os.path.exists(index_file):
        try:
            _vector_store = FAISS.load_local(
                _index_dir, _get_local_embeddings(), allow_dangerous_deserialization=True
            )
            return True
        except Exception:
            pass
    return False


def get_retriever():
    """获取 FAISS 检索器（top_k=4）"""
    if _vector_store is None:
        _load_index()
    if _vector_store is None:
        return None
    return _vector_store.as_retriever(search_kwargs={"k": 4})


def reset_index():
    """清空索引"""
    global _vector_store
    _vector_store = None
    import shutil
    if os.path.isdir(_index_dir):
        shutil.rmtree(_index_dir, ignore_errors=True)


def index_stats() -> Dict:
    """返回当前索引状态"""
    return {
        "has_index": _vector_store is not None,
        "index_dir": _index_dir,
    }


def index_sample_docs():
    """自动索引 data/sample_docs/ 下的示例文档（应用启动时调用）"""
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_docs")
    files = []
    if os.path.isdir(sample_dir):
        for ext in ("*.md", "*.json"):
            files.extend(glob.glob(os.path.join(sample_dir, ext)))
    if files:
        build_index_from_files(files)

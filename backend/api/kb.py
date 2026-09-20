"""API 路由 — 知识库管理"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import tempfile, os, json

from rag.indexer import build_index_from_files, reset_index, index_stats, get_retriever
from rag.retriever import retrieve_with_sources

router = APIRouter(prefix="/api/kb", tags=["kb"])

# 临时上传目录
_UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "rag_kb_uploads")
os.makedirs(_UPLOAD_DIR, exist_ok=True)

class KBSearchRequest(BaseModel):
    q: str
    top_k: int = 4

class KBSearchResponse(BaseModel):
    results: list

@router.post("/import")
async def import_docs(files: list[UploadFile] = File(...)):
    """
    上传文档并导入知识库
    支持：.md / .json 文件
    """
    if not files:
        raise HTTPException(status_code=400, detail="请上传至少一个文件")

    saved_paths = []
    for f in files:
        content = await f.read()
        text = content.decode("utf-8", errors="ignore")
        save_path = os.path.join(_UPLOAD_DIR, f.filename or "upload")
        with open(save_path, "w", encoding="utf-8") as out:
            out.write(text)
        saved_paths.append(save_path)

    doc_count = build_index_from_files(saved_paths)
    return {
        "imported": [f.filename for f in files],
        "new_chunks": doc_count,
        "index_total": doc_count,
    }

@router.get("/search", response_model=KBSearchResponse)
async def kb_search(req: KBSearchRequest):
    """直接检索知识库（不经过 LLM）"""
    results = retrieve_with_sources(req.q, top_k=req.top_k)
    return KBSearchResponse(results=results)

@router.get("/status")
async def kb_status():
    """查看当前知识库状态"""
    return index_stats()

@router.delete("")
async def kb_reset():
    """清空知识库索引"""
    reset_index()
    # 清理上传文件
    for f in os.listdir(_UPLOAD_DIR):
        fp = os.path.join(_UPLOAD_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)
    return {"message": "知识库已清空"}

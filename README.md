# RAGFlow Agent — RAG 知识问答 + Agent 工具调用系统

基于 **Python + LangChain + FastAPI** 的 AI Agent 应用，将 RAG 知识库检索与大模型工具调用（Function Calling）结合，支持自然语言多轮对话。

## 功能特性

- **RAG 知识问答**：基于 FAISS 向量库，支持本地 Markdown/JSON 文档导入，检索增强生成
- **Agent 工具调用**：通过 Function Calling 让大模型自主决策调用外部工具（计算器、天气查询、知识库检索）
- **多轮对话**：支持上下文记忆（会话管理），保留最近 N 轮对话历史
- **FastAPI 后端**：RESTful API，支持流式响应（SSE）
- **简洁前端**：单页 Web 对话界面，支持消息气泡展示、工具调用过程可视化

## 技术栈

| 模块 | 技术 |
|------|------|
| Agent 框架 | LangChain 0.2+ |
| LLM | OpenAI / 兼容 API（如 DeepSeek、Moonshot 等） |
| 向量库 | FAISS (local) |
| 后端 | FastAPI + uvicorn |
| 前端 | Vanilla JS + HTML/CSS（单页） |
| 工具调用 | Function Calling (tool_calls) |

## 项目结构

```
rag_agent_project/
├── backend/
│   ├── main.py            # FastAPI 入口
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── core.py        # Agent 核心逻辑
│   │   ├── tools.py       # 工具定义（计算器、天气、RAG）
│   │   ├── llm.py         # LLM 封装（兼容 OpenAI API）
│   │   └── memory.py      # 会话记忆管理
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── indexer.py     # 文档索引构建
│   │   ├── retriever.py   # 检索器
│   │   └── chunker.py     # 文本分块
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py        # 对话接口
│   │   └── kb.py          # 知识库管理接口
│   └── requirements.txt
├── frontend/
│   ├── index.html         # 对话页面
│   ├── app.js
│   └── style.css
├── data/
│   └── sample_docs/       # 示例文档（Markdown）
└── tests/
    └── test_agent.py
```

## 快速开始

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 配置 API Key

在 `.env` 文件中添加（或设置环境变量）：

```
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.openai.com/v1   # 可换成 DeepSeek / Moonshot 等兼容 API
OPENAI_MODEL=gpt-4o-mini                      # 或 deepseek-chat / moonshot-v1-8k 等
```

### 3. 导入知识库文档（可选）

将 Markdown/JSON 文档放入 `data/sample_docs/`，启动时自动构建向量索引。

### 4. 启动后端

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 5. 打开前端

```bash
# 方式一：直接用浏览器打开
open ../frontend/index.html

# 方式二：启动静态服务
cd ../frontend && python -m http.server 3000
# 访问 http://localhost:3000
```

## API 说明

### 对话接口

```
POST /api/chat
{
  "session_id": "optional-uuid",
  "message": "帮我查一下北京今天的天气",
  "use_tools": true
}

Response:
{
  "session_id": "uuid",
  "reply": "...",
  "tool_calls": [
    {"tool": "get_weather", "args": {"city": "北京"}, "result": "..."}
  ],
  "sources": ["doc1.md", "doc2.md"]   # RAG 引用来源（如有）
}
```

### 流式对话（SSE）

```
POST /api/chat/stream
# 返回 Server-Sent Events，每块为 JSON 片段
```

### 知识库管理

```
POST   /api/kb/import     # 导入文档
GET    /api/kb/search?q=  # 直接检索
DELETE /api/kb            # 清空索引
```

## 工具说明

| 工具名 | 功能 | 触发场景 |
|--------|------|----------|
| `calculate` | 数学表达式计算 | 用户问数学题 |
| `get_weather` | 查询天气（模拟） | 用户问天气 |
| `search_knowledge_base` | RAG 知识库检索 | 用户问领域知识 |


"""
工具定义层 — Agent 可调用的外部工具（Function Calling）
使用 Pydantic 定义工具参数 schema，兼容 LangChain 0.2+
"""
import math
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

# ------------------------------------------------------------------
# 参数 Schema
# ------------------------------------------------------------------

class CalcArgs(BaseModel):
    expr: str = Field(description="数学表达式字符串，如 '2+3*4'、'sqrt(16)'、'sin(pi/2)'")

class WeatherArgs(BaseModel):
    city: str = Field(description="城市名称，如 '北京'、'广州'")

class KBAArgs(BaseModel):
    query: str = Field(description="检索关键词或问题")

# ------------------------------------------------------------------
# 1. 计算器工具
# ------------------------------------------------------------------

def _safe_eval(expr: str) -> str:
    allowed_names = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e, "sqrt": math.sqrt, "log": math.log,
    }
    try:
        safe = eval(expr, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return f"计算结果：{expr} = {safe}"
    except Exception as e:
        return f"计算错误：{str(e)}"

class CalculateTool(BaseTool):
    name: str = "calculate"
    description: str = "计算数学表达式。支持四则运算、幂运算、三角函数（sin/cos/tan/sqrt/log）。当用户问数学题或计算题时使用。"
    args_schema: type = CalcArgs

    def _run(self, expr: str) -> str:
        return _safe_eval(expr)

# ------------------------------------------------------------------
# 2. 天气查询工具（模拟数据）
# ------------------------------------------------------------------

_FAKE_WEATHER = {
    "北京": {"temp": "22°C", "condition": "晴", "humidity": "45%", "wind": "北风3级"},
    "上海": {"temp": "25°C", "condition": "多云", "humidity": "60%", "wind": "东南风2级"},
    "广州": {"temp": "28°C", "condition": "阵雨", "humidity": "78%", "wind": "南风2级"},
    "深圳": {"temp": "27°C", "condition": "多云", "humidity": "70%", "wind": "东风3级"},
    "杭州": {"temp": "24°C", "condition": "阴", "humidity": "55%", "wind": "西北风2级"},
    "成都": {"temp": "23°C", "condition": "小雨", "humidity": "80%", "wind": "微风"},
}

class GetWeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "查询指定城市的当前天气（温度、天气状况、湿度、风力）。当用户问天气、气温、要不要带伞等问题时使用。"
    args_schema: type = WeatherArgs

    def _run(self, city: str) -> str:
        data = _FAKE_WEATHER.get(city)
        if data:
            return f"{city}当前天气：{data['condition']}，气温 {data['temp']}，湿度 {data['humidity']}，{data['wind']}"
        return f"未找到城市「{city}」的天气数据，支持的城市：{'、'.join(_FAKE_WEATHER.keys())}"

# ------------------------------------------------------------------
# 3. RAG 知识库检索工具
# ------------------------------------------------------------------

class KnowledgeSearchTool(BaseTool):
    name: str = "search_knowledge_base"
    description: str = (
        "从本地知识库中检索相关文档片段。当用户问领域知识、文档内容、"
        "公司内部制度、产品说明等知识库中的信息时使用。"
    )
    args_schema: type = KBAArgs

    def _run(self, query: str) -> str:
        try:
            from rag.indexer import get_retriever
            retriever = get_retriever()
            if retriever is None:
                return "知识库为空或尚未初始化，请先通过 /api/kb/import 导入文档。"
            docs = retriever.invoke(query)
            if not docs:
                return f"未找到与「{query}」相关的知识库内容。"
            lines = [f"[来源: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs]
            return "\n---\n".join(lines)
        except Exception as e:
            return f"检索失败：{str(e)}"

# ------------------------------------------------------------------
# 工具列表
# ------------------------------------------------------------------

def get_all_tools():
    return [
        CalculateTool(),
        GetWeatherTool(),
        KnowledgeSearchTool(),
    ]

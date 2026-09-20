"""
基础测试 — 无需 LLM 即可运行的单元测试
运行：cd backend && python -m pytest ../tests/test_agent.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from agent.tools import CalculateTool, GetWeatherTool
from agent.memory import Session, SessionStore
from rag.chunker import chunk_markdown, chunk_documents


# ------------------------------------------------------------------
# 工具测试
# ------------------------------------------------------------------

def test_calculate_basic():
    tool = CalculateTool()
    assert "10" in tool._run("2+3*4")

def test_calculate_sqrt():
    tool = CalculateTool()
    result = tool._run("sqrt(16)")
    assert "4" in result

def test_calculate_error():
    tool = CalculateTool()
    result = tool._run("unknown_func(1)")
    assert "计算错误" in result

def test_weather_known_city():
    tool = GetWeatherTool()
    result = tool._run("广州")
    assert "广州" in result
    assert "°C" in result

def test_weather_unknown_city():
    tool = GetWeatherTool()
    result = tool._run("不存在的城市")
    assert "未找到" in result


# ------------------------------------------------------------------
# 会话测试
# ------------------------------------------------------------------

def test_session_basic():
    s = Session(session_id="test")
    s.add_user("你好")
    s.add_ai("你好！")
    assert len(s.messages) == 2
    assert s.messages[0].content == "你好"
    assert s.messages[1].content == "你好！"

def test_session_trim():
    s = Session(session_id="test2", max_turns=2)
    for i in range(5):
        s.add_user(f"msg{i}")
        s.add_ai(f"reply{i}")
    # max_turns=2 → 保留最近 4 条
    assert len(s.messages) == 4
    # 最早的已丢弃
    assert "msg0" not in [m.content for m in s.messages]


# ------------------------------------------------------------------
# 分块测试
# ------------------------------------------------------------------

def test_chunk_markdown():
    text = "# Title\n\n" + ("内容段落。" * 200)
    docs = chunk_markdown(text, "test.md")
    assert len(docs) >= 1
    assert all(d.metadata["source"] == "test.md" for d in docs)

def test_chunk_documents_routes_to_markdown():
    docs = chunk_documents("hello world", "a.txt")
    assert len(docs) >= 1

if __name__ == "__main__":
    print("运行所有测试...")
    for fn in [
        test_calculate_basic, test_calculate_sqrt, test_calculate_error,
        test_weather_known_city, test_weather_unknown_city,
        test_session_basic, test_session_trim,
        test_chunk_markdown, test_chunk_documents_routes_to_markdown,
    ]:
        fn()
        print(f"  ✅ {fn.__name__}")
    print("全部通过！")

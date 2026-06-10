from __future__ import annotations

from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()

# 세션별 메모리 저장소(단기 메모리)
SESSIONS: Dict[str, Dict[str, str]] = {}

def _get_session_store(session_id: str) -> Dict[str, str]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {}
    return SESSIONS[session_id]

@tool
def memory_put(session_id: str, key: str, value: str) -> str:
    """세션별 메모리에 key/value를 저장합니다."""
    store = _get_session_store(session_id)
    store[key] = value
    return f"saved:{session_id}:{key}"

@tool
def memory_get(session_id: str, key: str) -> str:
    """세션별 메모리에서 key 값을 조회합니다."""
    store = _get_session_store(session_id)
    return store.get(key, f"not_found:{session_id}:{key}")

TOOLS: List[BaseTool] = [memory_put, memory_get]
TOOL_REGISTRY: Dict[str, Any] = {t.name: t for t in TOOLS}

def execute_tool_call(tool_call: Dict[str, Any]) -> Any:
    name = tool_call.get("name")
    args = tool_call.get("args", {}) or {}
    if name not in TOOL_REGISTRY:
        raise ValueError(f"알 수 없는 도구입니다: {name}")
    if not isinstance(args, dict):
        raise TypeError("tool args는 dict 형태여야 합니다.")
    return TOOL_REGISTRY[name].invoke(args)

def run_turn_loop(llm_with_tools: Any, messages: List[Any], max_steps: int = 8) -> List[Any]:
    step = 0
    while True:
        step += 1
        if step > max_steps:
            messages.append(SystemMessage(content=f"[중단] 최대 스텝({max_steps})에 도달해 종료합니다."))
            return messages

        ai = llm_with_tools.invoke(messages)
        messages.append(ai)

        tool_calls = getattr(ai, "tool_calls", None) or []
        print(f"[step {step}] tool_calls:", tool_calls)

        if not tool_calls:
            return messages

        for tc in tool_calls:
            try:
                result = execute_tool_call(tc)
                print(f"[step {step}] tc[{tc['name']}] >>> {result}")
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            except Exception as e:
                err = f"tool_error:{type(e).__name__}:{e}"
                print(f"[step {step}] tc[{tc.get('name')}] >>> {err}")
                messages.append(ToolMessage(content=err, tool_call_id=tc["id"]))

def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    # 같은 코드로 세션만 바꿔가며 상태가 분리되는지 확인
    session_id = "user1"

    messages: List[Any] = [
        SystemMessage(content="사용자 정보는 memory_put/memory_get으로 저장/조회한 뒤 답변합니다."),
        HumanMessage(content=f"내 이름을 김범준으로 저장해줘. session_id={session_id}, key=name"),
    ]

    # ① memory_put 호출 유도(루프 실행)
    messages = run_turn_loop(llm_with_tools, messages, max_steps=6)
    print("[1턴 최종]")
    print(messages[-1].content)

    messages.append(HumanMessage(content=f"name 값을 조회해서 내 이름이 뭔지 말해줘. session_id={session_id}, key=name"))
    messages = run_turn_loop(llm_with_tools, messages, max_steps=6)
    print("[2턴 최종]")
    print(messages[-1].content)

    # 세션 변경 후 동일 key 조회(세션 분리 확인)
    other_session = "user2"
    messages2: List[Any] = [
        SystemMessage(content="사용자 정보는 memory_put/memory_get으로 저장/조회한 뒤 답변합니다."),
        HumanMessage(content=f"name 값을 조회해서 내 이름이 뭔지 말해줘. session_id={other_session}, key=name"),
    ]
    messages2 = run_turn_loop(llm_with_tools, messages2, max_steps=6)
    print("[user2 세션 최종]")
    print(messages2[-1].content)

if __name__ == "__main__":
    main()
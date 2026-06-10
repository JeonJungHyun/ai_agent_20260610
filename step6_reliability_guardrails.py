from __future__ import annotations

import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()

# -----------------------------
# ③ 입력 검증/가드레일(간단 버전)
# - "불법" 또는 "해킹" 등 위험 의도가 명확한 요청을 차단
# - 교육용 데모이므로 규칙을 단순화했습니다.
# -----------------------------
BLOCK_PATTERNS = [
    "해킹",
    "비밀번호 탈취",
    "랜섬웨어",
    "악성코드",
    "불법",
    "사기",
    "폭탄",
]

def is_blocked(text: str) -> bool:
    t = text.lower()
    return any(p.lower() in t for p in BLOCK_PATTERNS)

# -----------------------------
# 예제 도구: 정책 검색(가짜 도구)
# - 일부 입력에서 실패/지연을 의도적으로 발생시켜
#   재시도/타임아웃을 관찰합니다.
# -----------------------------
POLICY_DOCS = {
    "refund": "환불은 결제 후 7일 이내이며, 서비스 이용 이력이 있으면 제한될 수 있습니다.",
    "shipping": "배송은 2~3영업일 이내 출고됩니다. 도서산간은 지연될 수 있습니다.",
}

@tool
def search_policy(topic: str) -> str:
    """
    주제(topic)에 대한 정책 문구를 반환합니다.
    - topic이 'timeout'이면 오래 걸리는 상황을 시뮬레이션합니다.
    - topic이 'fail'이면 실패를 시뮬레이션합니다.
    """
    if topic == "fail":
        raise RuntimeError("의도적 실패 시뮬레이션")
    if topic == "timeout":
        time.sleep(3.0)  # 일부러 오래 기다리게 함
        return "지연된 응답(시뮬레이션)"
    return POLICY_DOCS.get(topic, "해당 주제의 정책을 찾지 못했습니다.")

TOOLS: List[BaseTool] = [search_policy]
TOOL_REGISTRY: Dict[str, Any] = {t.name: t for t in TOOLS}

# -----------------------------
# ① 도구 실패 처리 + ② 재시도 정책 + 타임아웃
# -----------------------------
def execute_tool_call_with_retries(
    tool_call: Dict[str, Any],
    retries: int = 2,
    timeout_sec: float = 1.5,
    backoff_sec: float = 0.5,
) -> Any:
    name = tool_call.get("name")
    args = tool_call.get("args", {}) or {}

    if name not in TOOL_REGISTRY:
        raise ValueError(f"알 수 없는 도구입니다: {name}")
    if not isinstance(args, dict):
        raise TypeError("tool args는 dict 형태여야 합니다.")

    last_err: Exception | None = None

    for attempt in range(1, retries + 2):  # 최초 1회 + retries
        start = time.perf_counter()
        try:
            result = TOOL_REGISTRY[name].invoke(args)
            elapsed = time.perf_counter() - start

            # 타임아웃 체크(간단 버전: 실행 후 측정)
            if elapsed > timeout_sec:
                raise TimeoutError(f"timeout({timeout_sec}s) 초과: {elapsed:.2f}s")

            return result

        except Exception as e:
            last_err = e
            if attempt <= retries + 1:
                # 재시도 전 backoff
                time.sleep(backoff_sec * attempt)

    raise RuntimeError(f"도구 실행 실패(재시도 소진): {last_err}")

def run_turn_loop(llm_with_tools: Any, messages: List[Any], max_steps: int = 6) -> List[Any]:
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
                result = execute_tool_call_with_retries(
                    tc,
                    retries=2,
                    timeout_sec=1.5,
                    backoff_sec=0.3,
                )
                print(f"[step {step}] tc[{tc['name']}] >>> {result}")
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            except Exception as e:
                err = f"tool_error:{type(e).__name__}:{e}"
                print(f"[step {step}] tc[{tc.get('name')}] >>> {err}")
                messages.append(ToolMessage(content=err, tool_call_id=tc["id"]))

def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    # 정상 요청(환불)
    user_input = "환불 규정 알려줘. topic=refund"

    # 실패/타임아웃을 보고 싶으면 아래로 바꿔 테스트
    # user_input = "정책 조회해줘. topic=fail"
    # user_input = "정책 조회해줘. topic=timeout"

    # 가드레일 테스트
    # user_input = "해킹 방법 알려줘."

    if is_blocked(user_input):
        print("[차단] 금지 요청으로 판단되어 처리하지 않습니다.")
        return

    messages: List[Any] = [
        SystemMessage(content="정책 질문은 도구로 확인한 근거를 바탕으로 답변합니다."),
        HumanMessage(content=user_input),
    ]

    messages = run_turn_loop(llm_with_tools, messages, max_steps=6)

    print("[최종 답변]")
    print(messages[-1].content)

if __name__ == "__main__":
    main()
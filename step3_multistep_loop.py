from __future__ import annotations

from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()

@tool
def calculate_vat(price: int) -> Dict[str, int]:
    """공급가액에 부가세(10%)를 계산해 총액을 반환합니다."""
    if price < 0:
        raise ValueError("price는 0 이상이어야 합니다.")
    vat = int(price * 0.1)
    total = price + vat
    return {"supply_price": price, "vat": vat, "total_price": total}

@tool
def apply_discount(price: int, rate: float) -> Dict[str, Any]:
    """가격에 할인율을 적용해 할인 후 금액을 반환합니다. rate는 0.0~1.0 범위입니다."""
    if price < 0:
        raise ValueError("price는 0 이상이어야 합니다.")
    if not (0.0 <= rate <= 1.0):
        raise ValueError("rate는 0.0~1.0 범위여야 합니다.")
    discounted = int(price * (1 - rate))
    return {"original_price": price, "discount_rate": rate, "discounted_price": discounted}

@tool
def format_krw(amount: int) -> str:
    """정수 금액(원)을 '1,234,567원' 형식 문자열로 변환합니다."""
    if amount < 0:
        raise ValueError("amount는 0 이상이어야 합니다.")
    return f"{amount:,}원"

TOOLS: List[BaseTool] = [calculate_vat, apply_discount, format_krw]
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
    """
    멀티스텝 루프:
    - LLM이 tool_calls를 만들면 실행하고 ToolMessage로 추가
    - tool_calls가 없으면 종료
    - max_steps로 무한 루프 방지
    """
    step = 0

    while True:
        step += 1
        if step > max_steps:
            messages.append(
                SystemMessage(content=f"[중단] 최대 스텝({max_steps})에 도달해 루프를 종료합니다.")
            )
            return messages

        ai = llm_with_tools.invoke(messages)
        messages.append(ai)

        tool_calls = getattr(ai, "tool_calls", None) or []
        print(f"[step {step}] tool_calls:", tool_calls)

        # ② 종료 조건: 도구 호출이 없으면 최종 답변으로 보고 종료
        if not tool_calls:
            return messages

        # 도구 호출이 있으면 모두 실행
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

    messages: List[Any] = [
        SystemMessage(content="가격 계산은 도구로 처리합니다. 필요하면 여러 번 도구를 호출해도 됩니다."),
        HumanMessage(content="정가 120000원에서 15% 할인하고, 부가세 포함 총액을 원화 표기로 알려줘."),
    ]

    # ① while 루프로 반복 실행 + ② 종료 조건 + ③ max_steps로 무한루프 방지
    messages = run_turn_loop(llm_with_tools, messages, max_steps=8)

    print("[최종 답변]")
    print(messages[-1].content)

if __name__ == "__main__":
    main()

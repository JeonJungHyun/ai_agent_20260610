from __future__ import annotations

from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()

@tool
def calculate_vat(price: int) -> Dict[str, int]:
    """
    공급가액을 입력 받아 부가세(10%)와 총액을 계산합니다.
    price는 정수(원 단위)입니다.
    """
    if price < 0 :
        raise ValueError("price는 0이상이어야 합니다.")
    
    vat = int(price * 0.1)
    total = price + vat
    return {
        "supply_price": price,
        "vat": vat,
        "total_price": total
    }
    
@tool
def apply_discount(price: int, rate: float) -> Dict[str, Any]:
    """
    가격에 할인율을 적용합니다.
    rate는 0.0~1.0 범위(예: 0.15는 15% 할인)입니다.
    """
    if price < 0:
        raise ValueError("price는 0이상이어야 합니다.")
    if not (0.0 <= rate <= 1.0 ):
        raise ValueError("rate는 0.0~1.0 범위여야 합니다.")
    discounted = int(price * (1 - rate))
    return {
        "supply_price": price,
        "discount_rate": rate,
        "discounted_price": discounted
    }

@tool
def format_krw(amount: int) -> str:
    """정수 금액(원)을 '1,234,567원' 형식의 문자열로 변환합니다."""
    if amount < 0:
        raise ValueError("amount는 0 이상이어야 합니다.")
    return f"{amount:,}원"


TOOLS: List[BaseTool] = [calculate_vat, apply_discount, format_krw]
TOOL_REGISTRY: Dict[str, Any] = {t.name: t for t in TOOLS}


def execute_tool_call(tool_call: Dict[str, Any]) -> Any:
    name = tool_call.get("name")
    args = tool_call.get("args", {}) or {}
    
    if name not in TOOL_REGISTRY:
        raise ValueError(f"알 수 없는 도구입니다:{name}")
    if not isinstance(args, dict):
        raise TypeError("tool args는 Dict여야 합니다.")
    
    return TOOL_REGISTRY[name].invoke(args)


def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)
    
    messages: List[Any] = [
        SystemMessage(content="가격 계산이 필요하면 반드시 도구를 호출해서 처리합니다. 여러도구가 필요하면 연쇄 호출 합니다."),
        HumanMessage(content="공급가액이 100000원일때 15%할인 된 금액의 부가세와 총액을 계산해줘."),
    ]

    ai = llm_with_tools.invoke(messages)
    messages.append(ai)
    
    print("[1] tool_calls")
    print(ai.tool_calls)
    
    for tc in ai.tool_calls:
        try:
            result = execute_tool_call(tc)
            print(f"[2] {tc['name']} >>> {result}")
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        except Exception as e:
            err = f"tool_error:{type(e).__name__}:{e}"
            print(f"[2-error] {tc['name']} >>> {err}")
            messages.append(ToolMessage(content=err, tool_call_id=tc["id"]))
            
        
    final = llm_with_tools.invoke(messages)
    
    print("[3] 최종 답변")
    print(final.content)
    

if __name__ == "__main__":
    main()

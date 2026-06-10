from __future__ import annotations
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
load_dotenv()


# @tool 데코레이터를 붙이면 이 함수는 LangChain에서 사용할 수 있는 "도구"가 됩니다.
# 즉, LLM이 필요하다고 판단하면 이 함수를 호출할 수 있습니다.
@tool
def calculate_vat(price: int) -> Dict[str, int]:
    """
    공급가액을 입력받아 부가세(10%)와 총액을 계산합니다.
    price는 정수(원 단위)입니다.
    """
    vat = int(price * 0.1)
    total = price + vat

    # 계산 결과를 딕셔너리 형태로 반환합니다.
    # LLM은 이 결과를 받아서 자연어 답변을 만들 수 있습니다.
    return {
        "supply_price": price,
        "vat": vat,
        "total_price": total
    }


# LLM에게 연결할 도구 목록입니다.
# 도구가 여러 개라면 이 리스트에 계속 추가하면 됩니다.
TOOLS: List[BaseTool] = [calculate_vat]


# 도구 이름으로 실제 도구 객체를 찾기 위한 딕셔너리입니다.
# 예: {"calculate_vat": calculate_vat 도구 객체}
#
# LLM이 tool_calls에서 "calculate_vat"라는 이름을 보내면,
# 이 딕셔너리를 이용해 실제 실행할 도구를 찾습니다.
TOOL_REGISTRY: Dict[str, Any] = {t.name: t for t in TOOLS}


def execute_tool_call(tool_call: Dict[str, Any]) -> Any:
    # LLM이 호출하려는 도구 이름을 가져옵니다.
    name = tool_call.get("name")

    # LLM이 도구에 전달하려는 인자 값을 가져옵니다.
    args = tool_call.get("args", {}) or {}

    # LLM이 요청한 도구 이름이 등록된 도구 목록에 없으면 오류를 발생시킵니다.
    if name not in TOOL_REGISTRY:
        raise ValueError(f"알 수 없는 도구입니다: {name}")

    # 도구 인자는 반드시 딕셔너리 형태여야 합니다.
    if not isinstance(args, dict):
        raise TypeError("Tool args는 Dict여야 합니다.")

		# 도구 호출
    # TOOL_REGISTRY[name]은 calculate_vat 도구 객체
    # invoke(args) ==> calculate_vat(price=100000)
    #
    # LangChain 도구는 내부적으로 입력 스키마 검증을 한 번 더 수행합니다.
    return TOOL_REGISTRY[name].invoke(args)


def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # LLM에 도구 목록을 연결.
    llm_with_tools = llm.bind_tools(TOOLS)

    # LLM에게 전달할 대화 메시지 목록.
    messages: List[Any] = [
        # SystemMessage는 LLM의 행동 규칙.
        # 가격 계산이 필요하면 반드시 도구를 사용하라고 지시.
        SystemMessage(content="가격 계산이 필요하면 반드시 도구를 호출해 처리합니다."),

        # HumanMessage는 사용자의 질문.
        HumanMessage(content="공급가액이 100000원일 때 부가세와 총액을 계산해줘."),
    ]

    # ① LLM에게 질문을 보냅니다.
    # 이때 LLM은 바로 최종 답변을 하지 않고,
    # calculate_vat 도구를 호출해야겠다고 판단.
    ai = llm_with_tools.invoke(messages)

    # LLM의 응답을 대화 기록에 추가.
    # 이 응답 안에는 tool_calls 정보가 들어 있을 수 있다.
    messages.append(ai)

    print("[1] tool_calls")

    # LLM이 어떤 도구를 호출하려고 했는지 출력.
    # 예: [{'name': 'calculate_vat', 'args': {'price': 100000}, ...}]
    print(ai.tool_calls)

    # ② LLM이 요청한 도구 호출을 직접 실행.
    for tc in ai.tool_calls:
        # tool_call 정보를 바탕으로 실제 도구 함수를 실행.
        result = execute_tool_call(tc)

        print(f"[2] tc[{tc['name']}] >>> {result}")

        # 도구 실행 결과를 ToolMessage로 다시 대화 기록에 추가.
        # tool_call_id는 LLM이 요청한 도구 호출과 실행 결과를 연결하는 ID.
        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tc["id"]
            )
        )

    # ③ 도구 실행 결과까지 포함된 메시지를 다시 LLM에게 전달.
    # 이제 LLM은 계산 결과를 바탕으로 최종 자연어 답변을 생성.
    final = llm_with_tools.invoke(messages)

    print("[3] 최종 답변")
    print(final.content)


if __name__ == "__main__":
    main()
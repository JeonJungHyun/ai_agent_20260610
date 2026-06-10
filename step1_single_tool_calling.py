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
    공급가액을 입력받아 부가세(10%)와 총액을 계산합니다.
    price는 정수(원 단위)입니다.
    """
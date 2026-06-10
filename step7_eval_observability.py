from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

# -----------------------------
# (A) 실행 로그 표준화: JSONL 스타일
# -----------------------------
def log_event(event: str, payload: Dict[str, Any]) -> None:
    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        **payload,
    }
    print(json.dumps(row, ensure_ascii=False))

# -----------------------------
# (B) RAG 검색 도구(5단계의 최소 버전)
# -----------------------------
RAW_DOCS: List[Dict[str, str]] = [
    {
        "id": "doc-01",
        "title": "환불 정책",
        "text": "스트리밍 시청 기록이 있는 경우 환불이 제한될 수 있습니다. 결제 후 7일 이내라도 서비스 이용 이력이 있으면 환불이 거절될 수 있습니다.",
    },
    {
        "id": "doc-02",
        "title": "배송 정책",
        "text": "주문 후 2~3영업일 이내 출고됩니다. 도서산간 지역은 배송이 지연될 수 있습니다.",
    },
]

_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=220, chunk_overlap=40)
_VECTORSTORE: Optional[FAISS] = None

def ensure_vectorstore() -> FAISS:
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    docs: List[Document] = []
    for d in RAW_DOCS:
        base = Document(page_content=d["text"], metadata={"id": d["id"], "title": d["title"]})
        docs.extend(_SPLITTER.split_documents([base]))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    _VECTORSTORE = FAISS.from_documents(docs, embeddings)
    return _VECTORSTORE

@tool
def search_docs(query: str, k: int = 2) -> List[Dict[str, str]]:
    """
    질의(query)로 문서를 검색해 상위 k개의 근거를 반환합니다.
    반환 형식: [{id, title, snippet}, ...]
    """
    vs = ensure_vectorstore()
    results = vs.similarity_search(query, k=k)
    out: List[Dict[str, str]] = []
    for doc in results:
        out.append(
            {
                "id": str(doc.metadata.get("id", "")),
                "title": str(doc.metadata.get("title", "")),
                "snippet": doc.page_content,
            }
        )
    return out

TOOLS: List[BaseTool] = [search_docs]
TOOL_REGISTRY: Dict[str, Any] = {t.name: t for t in TOOLS}

def execute_tool_call(tool_call: Dict[str, Any]) -> Any:
    name = tool_call.get("name")
    args = tool_call.get("args", {}) or {}
    if name not in TOOL_REGISTRY:
        raise ValueError(f"알 수 없는 도구입니다: {name}")
    return TOOL_REGISTRY[name].invoke(args)

# -----------------------------
# (C) 에이전트 루프(3단계 기반)
# -----------------------------
def run_turn_loop(llm_with_tools: Any, messages: List[Any], max_steps: int = 6) -> List[Any]:
    step = 0
    while True:
        step += 1
        if step > max_steps:
            log_event("loop_stop", {"reason": "max_steps", "max_steps": max_steps})
            messages.append(SystemMessage(content=f"[중단] 최대 스텝({max_steps})에 도달해 종료합니다."))
            return messages

        t0 = time.perf_counter()
        ai = llm_with_tools.invoke(messages)
        dt = time.perf_counter() - t0
        messages.append(ai)

        tool_calls = getattr(ai, "tool_calls", None) or []
        log_event("llm_response", {"step": step, "elapsed_sec": round(dt, 4), "tool_calls": tool_calls})

        if not tool_calls:
            log_event("loop_stop", {"reason": "no_tool_calls"})
            return messages

        for tc in tool_calls:
            t1 = time.perf_counter()
            try:
                result = execute_tool_call(tc)
                dt2 = time.perf_counter() - t1
                log_event("tool_ok", {"step": step, "tool": tc["name"], "elapsed_sec": round(dt2, 4)})
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            except Exception as e:
                dt2 = time.perf_counter() - t1
                err = f"tool_error:{type(e).__name__}:{e}"
                log_event("tool_fail", {"step": step, "tool": tc.get("name"), "elapsed_sec": round(dt2, 4), "error": err})
                messages.append(ToolMessage(content=err, tool_call_id=tc["id"]))

# -----------------------------
# (D) 케이스 기반 테스트 + 간단 자동 평가
# - 자동 평가는 "정답 정확도"까지 하려면 별도 기준이 필요하므로
#   여기서는 "근거 포함 여부"를 최소 통과 조건으로 둡니다.
# -----------------------------
@dataclass
class TestCase:
    name: str
    user_input: str
    must_include_titles: List[str]  # 답변에 포함되어야 하는 근거 제목(문자열 포함 체크)

def evaluate_answer(answer: str, must_include_titles: List[str]) -> Dict[str, Any]:
    missing = [t for t in must_include_titles if t not in answer]
    return {
        "pass": len(missing) == 0,
        "missing": missing,
    }

def run_test_case(llm_with_tools: Any, tc: TestCase) -> Dict[str, Any]:
    log_event("test_start", {"case": tc.name})

    messages: List[Any] = [
        SystemMessage(content="정책/규정 질문은 search_docs로 근거를 확보한 뒤 답변합니다. 답변에 근거 제목을 포함합니다."),
        HumanMessage(content=tc.user_input),
    ]

    messages = run_turn_loop(llm_with_tools, messages, max_steps=6)
    answer = messages[-1].content

    eval_result = evaluate_answer(answer, tc.must_include_titles)
    log_event("test_end", {"case": tc.name, "pass": eval_result["pass"], "missing": eval_result["missing"]})

    return {
        "case": tc.name,
        "answer": answer,
        "eval": eval_result,
    }

def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    test_cases: List[TestCase] = [
        TestCase(
            name="refund_policy_with_evidence",
            user_input="스트리밍을 조금 봤는데 환불 가능한가요? 근거도 같이 알려줘.",
            must_include_titles=["환불 정책"],
        ),
        TestCase(
            name="shipping_policy_with_evidence",
            user_input="배송은 보통 며칠 걸리나요? 근거도 같이 알려줘.",
            must_include_titles=["배송 정책"],
        ),
    ]

    results: List[Dict[str, Any]] = []
    for tc in test_cases:
        results.append(run_test_case(llm_with_tools, tc))

    print("\n[요약]")
    for r in results:
        print(f"- {r['case']}: pass={r['eval']['pass']} missing={r['eval']['missing']}")

if __name__ == "__main__":
    main()
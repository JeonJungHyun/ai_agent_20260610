from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from fastapi import Response

load_dotenv()

# -----------------------------
# ② 환경변수/키 관리 + 운영 설정(모델 교체/비용)
# - .env에서 MODEL_NAME을 바꾸면 모델 교체 가능
# -----------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "6"))
TOP_K = int(os.getenv("TOP_K", "2"))

# -----------------------------
# (A) RAG 문서(데모)
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
    {
        "id": "doc-03",
        "title": "계정 정책",
        "text": "계정 공유는 금지됩니다. 비밀번호 공유 또는 동시 접속이 확인되면 이용이 제한될 수 있습니다.",
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

# -----------------------------
# (B) 검색 도구
# -----------------------------
@tool
def search_docs(query: str, k: int = 2) -> List[Dict[str, str]]:
    """질의(query)로 문서를 검색해 상위 k개의 근거를 반환합니다."""
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
# (C) Agent 루프
# -----------------------------
def run_turn_loop(llm_with_tools: Any, messages: List[Any], max_steps: int) -> List[Any]:
    step = 0
    while True:
        step += 1
        if step > max_steps:
            messages.append(SystemMessage(content=f"[중단] 최대 스텝({max_steps})에 도달해 종료합니다."))
            return messages

        ai = llm_with_tools.invoke(messages)
        messages.append(ai)

        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            return messages

        for tc in tool_calls:
            try:
                result = execute_tool_call(tc)
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            except Exception as e:
                err = f"tool_error:{type(e).__name__}:{e}"
                messages.append(ToolMessage(content=err, tool_call_id=tc["id"]))

# -----------------------------
# (D) FastAPI 서비스
# -----------------------------
app = FastAPI(title="Step8 Agent Service", version="1.0.0")

class ChatRequest(BaseModel):
    question: str
    session_id: str = "demo"  # 세션 확장 여지(8단계에서는 최소만)

class ChatResponse(BaseModel):
    answer: str
    model: str
    elapsed_ms: int
    evidence: List[Dict[str, str]]  # search_docs 결과를 그대로 노출(근거 확인용)

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    llm_with_tools = llm.bind_tools(TOOLS)

    messages: List[Any] = [
        SystemMessage(content="정책/규정 질문은 search_docs로 근거를 확보한 뒤 답변합니다. 답변에 근거를 반영합니다."),
        HumanMessage(content=req.question),
    ]

    t0 = time.perf_counter()
    messages = run_turn_loop(llm_with_tools, messages, max_steps=MAX_STEPS)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    answer = messages[-1].content

    # 운영 편의를 위해 근거를 별도 추출(가장 마지막 search_docs ToolMessage를 파싱)
    evidence: List[Dict[str, str]] = []
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and "doc-" in m.content:
            try:
                evidence = eval(m.content)  # 데모용(실무에서는 json.loads로 안전하게 처리)
            except Exception:
                evidence = []
            break

    # return ChatResponse(answer=answer, model=MODEL_NAME, elapsed_ms=elapsed_ms, evidence=evidence)
    # PowerShell 터미널에서 한글 깨질 경우 아래 Response로 변경
    return Response(
        content=answer,
        media_type="text/plain; charset=utf-8"
    )

"""
실행
uvicorn step8_service_fastapi:app --reload --port 8000

테스트(파워셸)
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat -ContentType "application/json" -Body '{"question":"스트리밍을 조금 봤는데 환불 가능한가요? 근거도 주세요.","session_id":"u1"}'
"""
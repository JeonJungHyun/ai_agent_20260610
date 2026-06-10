from __future__ import annotations

from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

# ① 문서 로더(예제에서는 로컬 상수 문서로 대체)
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

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
)

_VECTORSTORE: FAISS | None = None

def build_vectorstore() -> FAISS:
    docs: List[Document] = []
    for d in RAW_DOCS:
        base = Document(
            page_content=d["text"],
            metadata={"id": d["id"], "title": d["title"]},
        )
        chunks = _SPLITTER.split_documents([base])
        docs.extend(chunks)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(docs, embeddings)

def ensure_vectorstore() -> FAISS:
    global _VECTORSTORE
    if _VECTORSTORE is None:
        _VECTORSTORE = build_vectorstore()
    return _VECTORSTORE

@tool
def search_docs(query: str, k: int = 3) -> List[Dict[str, str]]:
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

        # ② 종료 조건: 도구 호출이 없으면 종료
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

    messages: List[Any] = [
        SystemMessage(content="정책/규정 질문은 search_docs로 근거를 확보한 뒤 답변합니다. 근거를 함께 제시합니다."),
        HumanMessage(content="스트리밍을 조금 봤는데 환불 가능한가요? 근거도 같이 알려줘."),
    ]

    messages = run_turn_loop(llm_with_tools, messages, max_steps=6)

    print("[최종 답변]")
    print(messages[-1].content)

if __name__ == "__main__":
    main()
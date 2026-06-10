# ai_agent_20260610

LangChain과 OpenAI 모델을 사용해 Tool Calling 기반 AI Agent를 단계별로 구현한 실습 프로젝트입니다. 단일 도구 호출에서 시작해 다중 도구 라우팅, 반복 실행 루프, 세션 메모리, RAG 검색, 신뢰성 보강, 평가/관측, FastAPI 서비스화까지 순서대로 확장합니다.

## 프로젝트 구성

| 파일 | 내용 |
| --- | --- |
| `step1_single_tool_calling.py` | 부가세 계산 도구(`calculate_vat`)를 LLM이 호출하고 결과를 다시 답변에 반영하는 기본 예제 |
| `step2_multi_tool_routing.py` | 부가세 계산, 할인 적용, 원화 포맷팅 등 여러 도구를 등록하고 필요한 도구를 선택해 실행하는 예제 |
| `step3_multistep_loop.py` | LLM 응답에 `tool_calls`가 남아 있는 동안 반복 실행하는 Agent 루프 예제 |
| `step4_memory_tools_with_session.py` | 세션별 인메모리 저장소를 사용해 사용자 정보를 저장/조회하는 메모리 도구 예제 |
| `step5_rag_search_tool.py` | 로컬 문서를 임베딩하고 FAISS 벡터스토어로 검색하는 RAG 검색 도구 예제 |
| `step6_reliability_guardrails.py` | 입력 차단, 도구 실행 재시도, 타임아웃 처리 등 안정성 보강 예제 |
| `step7_eval_observability.py` | JSONL 형태 실행 로그, 테스트 케이스, 간단한 자동 평가를 추가한 관측성 예제 |
| `step8_service_fastapi.py` | RAG 검색 Agent를 FastAPI `/chat` 엔드포인트로 제공하는 서비스 예제 |
| `requirements.txt` | 프로젝트 실행에 필요한 Python 패키지 목록 |

## 실행 환경

- Python 3.10 이상 권장
- OpenAI API 키 필요
- Windows PowerShell 기준 예시를 포함합니다.

## 설치

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

RAG 및 FastAPI 예제(`step5`~`step8`) 실행 시 아래 패키지가 추가로 필요할 수 있습니다.

```powershell
pip install langchain-community langchain-text-splitters faiss-cpu fastapi uvicorn
```

## 환경 변수

프로젝트 루트에 `.env` 파일을 만들고 OpenAI API 키를 설정합니다.

```env
OPENAI_API_KEY=your_openai_api_key
```

`step8_service_fastapi.py`는 아래 환경 변수로 모델과 실행 옵션을 조정할 수 있습니다.

```env
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0
MAX_STEPS=6
TOP_K=2
```

## 단계별 실행

각 파일은 독립 실행 가능한 예제입니다.

```powershell
python step1_single_tool_calling.py
python step2_multi_tool_routing.py
python step3_multistep_loop.py
python step4_memory_tools_with_session.py
python step5_rag_search_tool.py
python step6_reliability_guardrails.py
python step7_eval_observability.py
```

## FastAPI 서비스 실행

```powershell
uvicorn step8_service_fastapi:app --reload --port 8000
```

상태 확인:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

채팅 요청:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/chat `
  -ContentType "application/json" `
  -Body '{"question":"환불 가능한가요? 근거도 함께 알려주세요.","session_id":"u1"}'
```

## 학습 흐름

1. `step1`: LLM이 도구 호출을 결정하고, Python 코드가 실제 도구를 실행합니다.
2. `step2`: 여러 도구 중 필요한 도구를 선택해 조합합니다.
3. `step3`: 한 번의 호출로 끝나지 않는 작업을 반복 루프로 처리합니다.
4. `step4`: 세션 ID별 메모리 저장소를 두어 사용자 상태를 분리합니다.
5. `step5`: 문서를 청크로 나누고 임베딩해 검색 근거를 제공합니다.
6. `step6`: 금지 입력, 실패 재시도, 타임아웃 같은 운영상 방어 장치를 추가합니다.
7. `step7`: 실행 로그와 테스트 케이스를 통해 Agent 동작을 관측하고 평가합니다.
8. `step8`: Agent를 API 서비스 형태로 감싸 외부에서 호출할 수 있게 합니다.

## 주의 사항

- `.env`에는 API 키가 포함되므로 Git에 커밋하지 않습니다. 현재 `.gitignore`에 `.env`가 제외되어 있습니다.
- 일부 파일의 한글 주석/문자열이 터미널 인코딩 설정에 따라 깨져 보일 수 있습니다. PowerShell에서 UTF-8 출력 설정을 맞추면 확인이 더 쉽습니다.
- `step4`의 메모리는 프로세스 메모리에 저장되므로 프로그램이 종료되면 사라집니다.
- `step5`~`step8`은 OpenAI 임베딩 API를 사용하므로 실행 시 API 호출 비용이 발생할 수 있습니다.

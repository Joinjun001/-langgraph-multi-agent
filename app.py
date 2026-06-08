from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# main.py에서 컴파일된 LangGraph app 임포트
from main import app as agent_app

load_dotenv()

# FastAPI 인스턴스 생성
app = FastAPI(
    title="Adaptive & Self-Corrective RAG Agent API",
    description="한양대학교 기말 프로젝트 - LangGraph 기반 자가 수정 RAG 에이전트 서비스",
    version="1.0.0",
    root_path="/rag"
)

# CORS 설정 (외부 자바스크립트 클라이언트 및 웹 UI 연동 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 스키마 정의
class QueryRequest(BaseModel):
    question: str

# 응답 스키마 정의
class QueryResponse(BaseModel):
    question: str
    generation: str
    documents_count: int
    loop_count: int

@app.get("/", response_class=HTMLResponse, summary="에이전트 웹 인터페이스")
def read_root():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Adaptive RAG Agent API</h1><p>Template not found.</p>"

@app.get("/health", summary="서버 헬스 체크")
def health_check():
    return {"status": "healthy", "service": "Adaptive RAG Agent API"}

@app.post("/query", response_model=QueryResponse, summary="질문 처리 및 에이전트 실행")
async def run_query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="질문(question)은 비어있을 수 없습니다.")
        
    try:
        print(f"\n[API Request] Received query: {request.question}")
        
        # LangGraph 에이전트 입력 구성
        inputs = {
            "question": request.question,
            "loop_count": 0,
            "documents": []
        }
        
        # 에이전트 동기 실행 (FastAPI 내부적으로 비동기 처리)
        result = agent_app.invoke(inputs)
        
        return QueryResponse(
            question=result.get("question", request.question),
            generation=result.get("generation", "답변을 생성하지 못했습니다."),
            documents_count=len(result.get("documents", [])),
            loop_count=result.get("loop_count", 0)
        )
    except Exception as e:
        print(f"[API Error] Exception during agent invocation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"에이전트 처리 중 서버 에러가 발생했습니다: {str(e)}"
        )

# 서버 수동 실행용 엔트리포인트 (python app.py)
if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 주소로 개방하여 홈서버 배포 시 외부 접근 가능하게 설정
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_chat_provider, get_db, get_embedder
from app.models import ChatMessage
from app.rag.agent import run_agent

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    tools_used: list[dict]


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    db=Depends(get_db),
    chat_provider=Depends(get_chat_provider),
    embedder=Depends(get_embedder),
):
    result = run_agent(req.message, chat_provider, db, embedder)
    db.add(ChatMessage(session_id=req.session_id, role="user", content=req.message))
    db.add(
        ChatMessage(
            session_id=req.session_id,
            role="assistant",
            content=result["answer"],
            tool_calls={"used": result["tools_used"]},
        )
    )
    db.commit()
    return ChatResponse(answer=result["answer"], tools_used=result["tools_used"])

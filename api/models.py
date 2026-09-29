from typing import Any, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    role: str = Field(..., description="Target role name (e.g., 'librarian')")
    message: str = Field(..., description="User message text")
    session_id: Optional[str] = Field(None, description="Optional session ID (leave empty for a fresh session)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "role": "librarian",
                "message": "What are the library timings?",
                "session_id": None,
            }
        }
    }


class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant response text")
    session_id: str = Field(..., description="Conversation session ID")
    sources: list[str] = Field(default_factory=list, description="List of reference or tool sources used")
    blocked: bool = Field(False, description="Whether guardrails intercepted the query")
    elapsed: Optional[float] = Field(None, description="Response latency in seconds")


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Server health status")
    roles: list[str] = Field(default_factory=list, description="List of deployed or available roles")


class RoleSummary(BaseModel):
    name: str
    description: str
    student_model: str
    teacher_mode: str
    created_at: Optional[str] = None


class RoleDetail(BaseModel):
    name: str
    description: str
    student_model: str
    teacher_mode: str
    created_at: Optional[str] = None
    questionnaire: Optional[dict[str, Any]] = None
    tools: list[str] = Field(default_factory=list, description="Available tool names")
    data_files: list[str] = Field(default_factory=list, description="Ingested data filenames")

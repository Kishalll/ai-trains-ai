import json
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import yaml

from api.models import ChatRequest, ChatResponse, HealthResponse, RoleDetail, RoleSummary
from inference.chat import ChatOrchestrator

router = APIRouter(prefix="/api/v1")

# Global orchestrator cache: role_name -> ChatOrchestrator
_orchestrators: dict[str, ChatOrchestrator] = {}
_roles_dir: Path = Path("roles")


def set_roles_dir(path: Path | str):
    global _roles_dir
    _roles_dir = Path(path)


def get_orchestrator(role_name: str) -> ChatOrchestrator:
    if role_name in _orchestrators:
        return _orchestrators[role_name]

    role_path = _roles_dir / role_name
    if not role_path.exists() or not (role_path / "role.yaml").exists():
        raise HTTPException(status_code=404, detail=f"Role '{role_name}' not found")

    try:
        orch = ChatOrchestrator(role_path)
        orch.warmup()
        _orchestrators[role_name] = orch
        return orch
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize role '{role_name}': {e}")


def get_available_roles() -> list[str]:
    if not _roles_dir.exists():
        return []
    roles = []
    for p in sorted(_roles_dir.iterdir()):
        if p.is_dir() and (p / "role.yaml").exists():
            roles.append(p.name)
    return roles


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    orchestrator = get_orchestrator(request.role)
    result = orchestrator.ask(request.message, session_id=request.session_id)
    return ChatResponse(
        response=result["response"],
        session_id=result["session_id"],
        sources=result.get("sources", []),
        blocked=result.get("blocked", False),
        elapsed=result.get("elapsed"),
    )


@router.post("/chat/stream")
def chat_stream_endpoint(request: ChatRequest):
    orchestrator = get_orchestrator(request.role)

    def event_stream():
        for chunk in orchestrator.ask_stream(request.message, session_id=request.session_id):
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/health", response_model=HealthResponse)
def health_endpoint():
    roles = get_available_roles()
    return HealthResponse(status="ok", roles=roles)


@router.get("/roles", response_model=list[RoleSummary])
def list_roles_endpoint():
    summaries = []
    for name in get_available_roles():
        cfg_path = _roles_dir / name / "role.yaml"
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            summaries.append(
                RoleSummary(
                    name=name,
                    description=cfg.get("description", ""),
                    student_model=cfg.get("student_model", ""),
                    teacher_mode=cfg.get("teacher_mode", ""),
                    created_at=cfg.get("created_at"),
                )
            )
        except Exception:
            continue
    return summaries


@router.get("/roles/{name}", response_model=RoleDetail)
def get_role_detail_endpoint(name: str):
    role_path = _roles_dir / name
    cfg_path = role_path / "role.yaml"
    if not role_path.exists() or not cfg_path.exists():
        raise HTTPException(status_code=404, detail=f"Role '{name}' not found")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Gather tools
    tools = []
    tools_path = role_path / "tools.yaml"
    if tools_path.exists():
        try:
            with open(tools_path, "r", encoding="utf-8") as f:
                t_data = yaml.safe_load(f) or {}
            tools = [t.get("name") for t in t_data.get("tools", []) if t.get("name")]
        except Exception:
            pass

    # Gather data files
    data_files = []
    data_dir = role_path / "data"
    if data_dir.exists():
        data_files = [f.name for f in sorted(data_dir.iterdir()) if f.is_file()]

    return RoleDetail(
        name=name,
        description=cfg.get("description", ""),
        student_model=cfg.get("student_model", ""),
        teacher_mode=cfg.get("teacher_mode", ""),
        created_at=cfg.get("created_at"),
        questionnaire=cfg.get("questionnaire"),
        tools=tools,
        data_files=data_files,
    )

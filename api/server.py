import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import _orchestrators, get_orchestrator, router, set_roles_dir


def create_app(roles_dir: Path | str = "roles", prewarm_role: str | None = None) -> FastAPI:
    resolved_roles_dir = Path(roles_dir).resolve()
    target_role = prewarm_role or os.environ.get("AI_INSTITUTE_PREWARM_ROLE")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 1. Startup: Pre-warm target role (loads embedding weights & sends keep_alive: -1 to Ollama)
        if target_role:
            try:
                orch = get_orchestrator(target_role)
                orch.warmup()
            except Exception as e:
                print(f"Warning: Could not prewarm role '{target_role}': {e}")
        yield
        # 2. Shutdown: Release all cached orchestrator models from Ollama RAM
        for role_name, orch in list(_orchestrators.items()):
            try:
                orch.unload()
            except Exception:
                pass

    app = FastAPI(
        title="AI-Institute API",
        version="1.0.0",
        description="REST API for role-locked college AI assistants",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    set_roles_dir(resolved_roles_dir)
    app.include_router(router)

    return app


# Default app instance for ASGI servers
app = create_app(
    roles_dir=os.environ.get("AI_INSTITUTE_ROLES_DIR", "roles"),
    prewarm_role=os.environ.get("AI_INSTITUTE_PREWARM_ROLE"),
)

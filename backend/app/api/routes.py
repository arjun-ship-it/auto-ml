import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional

from app.api.schemas import (
    ProjectCreate, ProjectResponse,
    DataUploadResponse, DatabaseConnectionCreate, DatabaseConnectionResponse,
    PredictionRequest, PredictionResponse,
)
from app.data.connectors import save_uploaded_file, list_datasets
from app.agent import AutoMLAgent

router = APIRouter()


# --- Project endpoints ---

@router.post("/projects", response_model=ProjectResponse)
async def create_project(project: ProjectCreate):
    """Create a new AutoML project."""
    from app.db.database import get_db
    from app.db.models import Project
    from datetime import datetime

    project_id = str(uuid.uuid4())[:8]

    async for db in get_db():
        db_project = Project(
            id=project_id,
            name=project.name,
            description=project.description,
            created_at=datetime.utcnow(),
        )
        db.add(db_project)
        await db.commit()

    return ProjectResponse(
        id=project_id,
        name=project.name,
        description=project.description,
        created_at=datetime.utcnow(),
    )


@router.get("/projects")
async def list_projects():
    """List all projects."""
    from app.db.database import get_db
    from app.db.models import Project
    from sqlalchemy import select

    async for db in get_db():
        result = await db.execute(select(Project).order_by(Project.created_at.desc()))
        projects = result.scalars().all()
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "created_at": p.created_at.isoformat(),
            }
            for p in projects
        ]


@router.get("/projects/{project_id}")
async def get_project(project_id: str):
    """Get project details."""
    agent = AutoMLAgent(project_id)
    await agent.initialize()
    return await agent.get_project_summary()


# --- Data endpoints ---

@router.post("/projects/{project_id}/upload", response_model=DataUploadResponse)
async def upload_data(project_id: str, file: UploadFile = File(...)):
    """Upload a dataset file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    allowed_extensions = [".csv", ".xlsx", ".xls", ".parquet", ".json"]
    ext = "." + file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}",
        )

    content = await file.read()
    result = await save_uploaded_file(content, file.filename, project_id)

    return DataUploadResponse(**result)


@router.get("/projects/{project_id}/datasets")
async def get_datasets(project_id: str):
    """List datasets for a project."""
    return await list_datasets(project_id)


# --- Database Connection endpoints ---

@router.post("/projects/{project_id}/connections", response_model=DatabaseConnectionResponse)
async def create_connection(project_id: str, connection: DatabaseConnectionCreate):
    """Add a database connection."""
    from app.db.database import get_db
    from app.db.models import DatabaseConnection

    conn_id = str(uuid.uuid4())[:8]

    async for db in get_db():
        db_conn = DatabaseConnection(
            id=conn_id,
            project_id=project_id,
            name=connection.name,
            connection_string=connection.connection_string,
            db_type=connection.db_type,
        )
        db.add(db_conn)
        await db.commit()

    return DatabaseConnectionResponse(
        id=conn_id,
        name=connection.name,
        db_type=connection.db_type,
        connected=True,
    )


# --- Chat endpoint (non-WebSocket fallback) ---

@router.post("/projects/{project_id}/chat")
async def chat(project_id: str, message: dict):
    """Send a message to the agent (REST fallback)."""
    agent = AutoMLAgent(project_id)
    await agent.initialize()

    responses = []
    async for chunk in agent.process_message(message["message"]):
        responses.append(chunk)

    return {"responses": responses}

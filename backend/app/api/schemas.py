from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime


# --- Project ---
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    created_at: datetime
    message_count: int = 0


# --- Chat ---
class ChatMessage(BaseModel):
    message: str
    project_id: str


class ChatResponse(BaseModel):
    type: str  # "text", "tool_start", "tool_result"
    content: Optional[str] = None
    tool: Optional[str] = None
    result: Optional[Any] = None


# --- Data ---
class DataUploadResponse(BaseModel):
    dataset_id: str
    rows: int
    columns: int
    column_names: list[str]
    preview: list[dict]


class DatabaseConnectionCreate(BaseModel):
    name: str
    connection_string: str
    db_type: str  # postgres, mysql, sqlite


class DatabaseConnectionResponse(BaseModel):
    id: str
    name: str
    db_type: str
    connected: bool


# --- ML ---
class ModelInfo(BaseModel):
    model_id: str
    model_type: str
    problem_type: str
    metrics: Optional[dict] = None
    created_at: datetime


class PredictionRequest(BaseModel):
    model_id: str
    input_data: Optional[dict] = None
    prediction_horizon: Optional[int] = None


class PredictionResponse(BaseModel):
    predictions: list[Any]
    confidence_intervals: Optional[dict] = None
    model_id: str

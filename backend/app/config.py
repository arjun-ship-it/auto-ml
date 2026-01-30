from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App
    APP_NAME: str = "AutoML Agent"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Gemini API
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"
    MAX_TOKENS: int = 4096

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./automl.db"

    # File Storage
    UPLOAD_DIR: str = "./uploads"
    MODELS_DIR: str = "./trained_models"

    # Docker Execution
    DOCKER_ENABLED: bool = True
    DOCKER_IMAGE: str = "automl-sandbox:latest"
    EXECUTION_TIMEOUT: int = 300  # seconds

    # SQL Database Connections (user-provided)
    MAX_SQL_CONNECTIONS: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

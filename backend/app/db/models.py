from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.orm import relationship

from app.db.database import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(String(8), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    conversations = relationship("Conversation", back_populates="project", cascade="all, delete-orphan")
    memory = relationship("ProjectMemory", back_populates="project", uselist=False, cascade="all, delete-orphan")
    connections = relationship("DatabaseConnection", back_populates="project", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(8), ForeignKey("projects.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant
    content = Column(Text, nullable=False)  # JSON-encoded message content
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="conversations")


class ProjectMemory(Base):
    __tablename__ = "project_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(8), ForeignKey("projects.id"), unique=True, nullable=False)
    memory_data = Column(Text, nullable=False)  # JSON-encoded memory
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="memory")


class DatabaseConnection(Base):
    __tablename__ = "database_connections"

    id = Column(String(8), primary_key=True)
    project_id = Column(String(8), ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    connection_string = Column(Text, nullable=False)
    db_type = Column(String(50), nullable=False)  # postgres, mysql, sqlite
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="connections")

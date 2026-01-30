from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def init_db():
    """Initialize the database and create all tables."""
    async with engine.begin() as conn:
        from app.db.models import Project, Conversation, ProjectMemory, DatabaseConnection
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Get database session."""
    async with async_session() as session:
        yield session

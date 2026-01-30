import json
from typing import Any
from app.db.database import get_db
from app.db.models import Conversation


class ConversationManager:
    """Manages conversation history for a project."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.messages: list[dict[str, Any]] = []

    async def load_history(self):
        """Load conversation history from database."""
        async for db in get_db():
            from sqlalchemy import select
            stmt = select(Conversation).where(
                Conversation.project_id == self.project_id
            ).order_by(Conversation.created_at)
            result = await db.execute(stmt)
            records = result.scalars().all()

            self.messages = []
            for record in records:
                self.messages.append({
                    "role": record.role,
                    "content": json.loads(record.content)
                    if isinstance(record.content, str)
                    else record.content,
                })

    def add_message(self, role: str, content: Any):
        """Add a message to the conversation."""
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        self.messages.append({"role": role, "content": content})

    def add_tool_result(self, tool_use_id: str, result: Any):
        """Add a tool result message."""
        self.messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": json.dumps(result) if not isinstance(result, str) else result,
            }],
        })

    def get_messages(self) -> list[dict]:
        """Get formatted messages for Claude API."""
        return self.messages

    async def save_history(self):
        """Persist conversation to database."""
        async for db in get_db():
            for msg in self.messages:
                record = Conversation(
                    project_id=self.project_id,
                    role=msg["role"],
                    content=json.dumps(msg["content"])
                    if not isinstance(msg["content"], str)
                    else msg["content"],
                )
                db.add(record)
            await db.commit()

    def get_last_n_messages(self, n: int) -> list[dict]:
        """Get the last N messages."""
        return self.messages[-n:]

    def clear(self):
        """Clear conversation history."""
        self.messages = []

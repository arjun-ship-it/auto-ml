import json
from typing import Any, Optional
from app.db.database import get_db
from app.db.models import ProjectMemory


class MemoryManager:
    """Manages persistent memory for a project - stores key facts, decisions, and context."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.facts: dict[str, Any] = {}
        self.decisions: list[dict] = []
        self.data_profile: Optional[dict] = None
        self.model_history: list[dict] = []

    async def load(self):
        """Load memory from database."""
        async for db in get_db():
            from sqlalchemy import select
            stmt = select(ProjectMemory).where(
                ProjectMemory.project_id == self.project_id
            )
            result = await db.execute(stmt)
            record = result.scalar_one_or_none()

            if record:
                memory_data = json.loads(record.memory_data)
                self.facts = memory_data.get("facts", {})
                self.decisions = memory_data.get("decisions", [])
                self.data_profile = memory_data.get("data_profile")
                self.model_history = memory_data.get("model_history", [])

    async def save(self):
        """Persist memory to database."""
        memory_data = json.dumps({
            "facts": self.facts,
            "decisions": self.decisions,
            "data_profile": self.data_profile,
            "model_history": self.model_history,
        })

        async for db in get_db():
            from sqlalchemy import select
            stmt = select(ProjectMemory).where(
                ProjectMemory.project_id == self.project_id
            )
            result = await db.execute(stmt)
            record = result.scalar_one_or_none()

            if record:
                record.memory_data = memory_data
            else:
                record = ProjectMemory(
                    project_id=self.project_id,
                    memory_data=memory_data,
                )
                db.add(record)
            await db.commit()

    async def update(self, user_message: str, assistant_response: Any):
        """Update memory based on conversation turn."""
        # Extract key facts from the conversation
        if isinstance(assistant_response, list):
            for block in assistant_response:
                if block.get("type") == "text":
                    text = block.get("text", "")
                    # Store significant decisions
                    if any(keyword in text.lower() for keyword in [
                        "recommend", "selected", "best model", "chosen"
                    ]):
                        self.decisions.append({
                            "context": user_message[:100],
                            "decision": text[:200],
                        })
        await self.save()

    async def get_relevant_context(self, query: str) -> str:
        """Get relevant context for a query."""
        context_parts = []

        if self.facts:
            context_parts.append(f"Known Facts: {json.dumps(self.facts, indent=2)}")

        if self.data_profile:
            context_parts.append(
                f"Data Profile Summary: {json.dumps(self.data_profile, indent=2)}"
            )

        if self.model_history:
            context_parts.append(
                f"Model History: {json.dumps(self.model_history[-3:], indent=2)}"
            )

        if self.decisions:
            context_parts.append(
                f"Past Decisions: {json.dumps(self.decisions[-5:], indent=2)}"
            )

        return "\n\n".join(context_parts) if context_parts else ""

    async def get_summary(self) -> dict:
        """Get a summary of stored memory."""
        return {
            "facts_count": len(self.facts),
            "decisions_count": len(self.decisions),
            "has_data_profile": self.data_profile is not None,
            "models_tried": len(self.model_history),
        }

    def store_fact(self, key: str, value: Any):
        """Store a key fact about the project."""
        self.facts[key] = value

    def store_data_profile(self, profile: dict):
        """Store the data profile."""
        self.data_profile = profile

    def store_model_result(self, model_info: dict):
        """Store a model training result."""
        self.model_history.append(model_info)

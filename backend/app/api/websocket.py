import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.agent import AutoMLAgent

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, project_id: str):
        await websocket.accept()
        if project_id not in self.active_connections:
            self.active_connections[project_id] = []
        self.active_connections[project_id].append(websocket)

    def disconnect(self, websocket: WebSocket, project_id: str):
        if project_id in self.active_connections:
            self.active_connections[project_id].remove(websocket)
            if not self.active_connections[project_id]:
                del self.active_connections[project_id]

    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_json(data)


manager = ConnectionManager()


@router.websocket("/chat/{project_id}")
async def websocket_chat(websocket: WebSocket, project_id: str):
    """WebSocket endpoint for real-time chat with the agent."""
    await manager.connect(websocket, project_id)

    # Initialize agent
    agent = AutoMLAgent(project_id)
    await agent.initialize()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            user_message = message.get("message", "")

            if not user_message:
                await manager.send_json(websocket, {
                    "type": "error",
                    "content": "Empty message",
                })
                continue

            # Send acknowledgment
            await manager.send_json(websocket, {
                "type": "status",
                "content": "processing",
            })

            # Process message and stream responses
            async for chunk in agent.process_message(user_message):
                await manager.send_json(websocket, chunk)

            # Send completion signal
            await manager.send_json(websocket, {
                "type": "done",
                "content": "Message processed",
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket, project_id)
    except Exception as e:
        await manager.send_json(websocket, {
            "type": "error",
            "content": str(e),
        })
        manager.disconnect(websocket, project_id)

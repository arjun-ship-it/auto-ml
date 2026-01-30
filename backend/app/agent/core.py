import json
from typing import AsyncGenerator
from google import genai
from google.genai import types

from app.config import settings
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.conversation import ConversationManager
from app.agent.tools import GEMINI_TOOLS, execute_tool
from app.agent.memory import MemoryManager


class AutoMLAgent:
    """Core agent that orchestrates the AutoML pipeline using Gemini."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.conversation = ConversationManager(project_id)
        self.memory = MemoryManager(project_id)
        self.chat_history = []

    async def initialize(self):
        """Load conversation history and memory for the project."""
        await self.conversation.load_history()
        await self.memory.load()
        self.chat_history = self._convert_history_to_gemini_format()

    def _convert_history_to_gemini_format(self):
        """Convert stored conversation history to Gemini format."""
        gemini_history = []
        for msg in self.conversation.messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                gemini_history.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(content if isinstance(content, str) else str(content))]
                ))
            elif role == "assistant":
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if item.get("type") == "text":
                            parts.append(types.Part.from_text(item["text"]))
                    if parts:
                        gemini_history.append(types.Content(role="model", parts=parts))
                else:
                    gemini_history.append(types.Content(
                        role="model",
                        parts=[types.Part.from_text(content if isinstance(content, str) else str(content))]
                    ))
        
        return gemini_history

    async def process_message(self, user_message: str) -> AsyncGenerator[dict, None]:
        """Process a user message and yield response chunks."""
        self.conversation.add_message("user", user_message)

        context = await self.memory.get_relevant_context(user_message)
        
        full_message = user_message
        if context:
            full_message = f"{user_message}\n\n[Context: {context}]"

        self.chat_history.append(types.Content(
            role="user",
            parts=[types.Part.from_text(full_message)]
        ))

        response = await self.client.aio.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=self.chat_history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=GEMINI_TOOLS,
                max_output_tokens=settings.MAX_TOKENS,
            ),
        )

        assistant_content = []
        
        for part in response.candidates[0].content.parts:
            if part.text:
                assistant_content.append({"type": "text", "text": part.text})
                yield {"type": "text", "content": part.text}

            elif part.function_call:
                func_call = part.function_call
                tool_name = func_call.name
                tool_input = dict(func_call.args) if func_call.args else {}

                yield {
                    "type": "tool_start",
                    "tool": tool_name,
                    "input": tool_input,
                }

                tool_result = await execute_tool(
                    tool_name, tool_input, self.project_id
                )

                yield {
                    "type": "tool_result",
                    "tool": tool_name,
                    "result": tool_result,
                }

                assistant_content.append({
                    "type": "tool_use",
                    "name": tool_name,
                    "input": tool_input,
                })

                self.chat_history.append(types.Content(
                    role="model",
                    parts=[part]
                ))
                
                self.chat_history.append(types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(
                        name=tool_name,
                        response={"result": json.dumps(tool_result) if isinstance(tool_result, dict) else str(tool_result)}
                    )]
                ))

                follow_up = await self.client.aio.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=self.chat_history,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=GEMINI_TOOLS,
                        max_output_tokens=settings.MAX_TOKENS,
                    ),
                )

                for follow_part in follow_up.candidates[0].content.parts:
                    if follow_part.text:
                        yield {"type": "text", "content": follow_part.text}
                        assistant_content = [{"type": "text", "text": follow_part.text}]
                        self.chat_history.append(types.Content(
                            role="model",
                            parts=[follow_part]
                        ))

        if assistant_content:
            self.conversation.add_message("assistant", assistant_content)

        await self.conversation.save_history()
        await self.memory.update(user_message, assistant_content)

    async def get_project_summary(self) -> dict:
        """Get a summary of the current project state."""
        return {
            "project_id": self.project_id,
            "message_count": len(self.conversation.messages),
            "memory": await self.memory.get_summary(),
        }

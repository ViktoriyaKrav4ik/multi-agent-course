from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from config import SYSTEM_PROMPT, Settings
from tools import TOOL_REGISTRY, tool_definitions


@dataclass
class ResearchAgent:
    settings: Settings = field(default_factory=Settings)
    client: OpenAI | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = OpenAI(api_key=self.settings.api_key.get_secret_value())
        if not self.messages:
            self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def run(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})

        tools = tool_definitions()
        max_iters = self.settings.max_iterations

        for step in range(1, max_iters + 1):
            response = self.client.chat.completions.create(
                model=self.settings.model_name,
                messages=self.messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
            )

            msg = response.choices[0].message

            # Always append the assistant message (even if it only contains tool calls)
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if msg.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]

            self.messages.append(assistant_message)

            tool_calls = msg.tool_calls or []
            if not tool_calls:
                return (msg.content or "").strip()

            for tc in tool_calls:
                name = tc.function.name
                raw_args = tc.function.arguments or "{}"

                print(f"\n🔧 Tool call: {name}(args={raw_args})")

                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except Exception as e:
                    result = f"ToolError: invalid JSON arguments for {name}: {e}"
                    print(f"📎 Result: {result}")
                    self.messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result}
                    )
                    continue

                fn = TOOL_REGISTRY.get(name)
                if fn is None:
                    result = f"ToolError: unknown tool: {name}"
                    print(f"📎 Result: {result}")
                    self.messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result}
                    )
                    continue

                try:
                    result = fn(**args)
                except Exception as e:
                    result = f"ToolError: {name} raised: {e}"

                # Keep tool outputs bounded to protect context
                if isinstance(result, str) and len(result) > 12000:
                    result = result[:12000] + "\n\n[... truncated tool output ...]"

                print(f"📎 Result: {result[:5000]}")

                self.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
                )

        return (
            "I reached the maximum number of iterations without a final answer. "
            "Try a narrower question or ask me to summarize what I found so far."
        )


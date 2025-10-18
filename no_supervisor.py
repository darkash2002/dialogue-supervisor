"""
Mode 1: No Supervisor
Baseline help assistant with web search, no supervision.
"""

import os
import asyncio
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def create_help_assistant():
    """Create a help assistant without any supervision."""
    with open("prompts/help_assistant.txt", "r") as f:
        instructions = f.read()

    agent = Agent(
        name="HelpAssistant",
        instructions=instructions,
        model="gpt-4.1-nano",
    )
    return agent


async def run_conversation():
    """Run a continuous conversation with the help assistant with streaming."""
    agent = create_help_assistant()
    messages = []

    while True:
        user_input = input("\n> ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        result = Runner.run_streamed(agent, messages)
        assistant_response = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
                assistant_response += event.data.delta

        messages.append({"role": "assistant", "content": assistant_response})
        print()  # New line at end


if __name__ == "__main__":
    asyncio.run(run_conversation())

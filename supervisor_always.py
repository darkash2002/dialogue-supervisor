import os
import asyncio
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent

try:
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
except Exception:
    raise Exception("Failed to load OpenAI api key.")

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class SupervisorResponse(BaseModel):
    """Structured supervisor response."""
    intervene: bool
    developer_message: str | None  # Only if intervene=True


# Session state management
active_monitor_tasks = {}  # session_id -> asyncio.Task
conversation_histories = {}  # session_id -> list of messages
conversation_versions = {}  # session_id -> int (version counter)
help_agents = {}  # session_id -> Agent
pending_dev_messages = {}  # session_id -> list of developer messages


async def analyze_with_supervisor(session_id: str) -> Optional[str]:
    """Analyze conversation using structured outputs with last 10 messages only."""
    conversation_history = conversation_histories.get(session_id, [])

    # Only use last 10 messages to reduce tokens
    recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history

    # Load supervisor prompt
    with open("prompts/supervisor.txt", "r") as f:
        instructions = f.read()

    # Format conversation
    conversation_text = "\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in recent_history
    ])

    # Use structured outputs API directly
    response = await openai_client.beta.chat.completions.parse(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"Analyze:\n{conversation_text}"}
        ],
        response_format=SupervisorResponse,
        reasoning_effort="low",
        max_completion_tokens=1000  # Allow space for reasoning + response
    )

    result = response.choices[0].message.parsed
    if result.intervene and result.developer_message:
        return result.developer_message
    return None


async def continuous_monitor_loop(session_id: str):
    """
    Continuous monitoring loop for always-on supervisor:
    1. Wait for conversation version change
    2. Analyze current state
    3. Queue developer message if needed
    4. If conversation changed during analysis, re-analyze
    """
    last_processed_version = -1

    while True:
        try:
            current_version = conversation_versions.get(session_id, 0)

            # Wait for conversation change
            if current_version == last_processed_version:
                await asyncio.sleep(0.1)
                continue

            # Analyze conversation with current version
            version_at_start = current_version
            dev_message = await analyze_with_supervisor(session_id)

            # Queue developer message if provided
            if dev_message:
                pending_dev_messages[session_id].append(dev_message)
                print(f"\n[Supervisor guidance ready for next turn]")

            # Check if conversation changed during analysis
            if conversation_versions.get(session_id, 0) > version_at_start:
                # Conversation changed - immediately re-analyze with latest state
                continue

            # Mark this version as processed
            last_processed_version = current_version

        except Exception as e:
            print(f"\n[Supervisor analysis failed: {e}]")
            # Continue monitoring despite error
            last_processed_version = conversation_versions.get(session_id, 0)


def start_continuous_monitoring(session_id: str):
    """Start continuous monitoring task for session."""
    if session_id not in active_monitor_tasks:
        task = asyncio.create_task(continuous_monitor_loop(session_id))
        active_monitor_tasks[session_id] = task


def on_conversation_change(session_id: str, new_messages: list):
    """Called when conversation has new messages (user or assistant)."""
    # Add messages to history
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    conversation_histories[session_id].extend(new_messages)

    # Increment version to trigger re-analysis
    conversation_versions[session_id] = conversation_versions.get(session_id, 0) + 1


def create_help_assistant(session_id: str):
    """Create help assistant for always-on supervision mode."""
    with open("prompts/help_assistant.txt", "r") as f:
        instructions = f.read()

    agent = Agent(
        name="HelpAssistant",
        instructions=instructions,
        model="gpt-4.1-nano",
    )
    return agent


async def run_conversation_with_always_on_supervisor(session_id: str):
    """
    Run conversation with always-on supervisor.
    Supervisor runs from the start, continuously monitoring.
    """
    conversation_histories[session_id] = []
    conversation_versions[session_id] = 0
    pending_dev_messages[session_id] = []
    conversation_context = []

    # Start continuous monitoring from the beginning
    start_continuous_monitoring(session_id)

    print("Welcome to the demo. How can we help you creating a food or workout plan? Type /exit to quit.")
    while True:
        user_input = input("\n> ")
        if user_input.lower() in ["/exit"]:
            print("Goodbye!")
            break

        # Add user message to history and trigger supervisor
        on_conversation_change(session_id, [{
            "role": "user",
            "content": user_input
        }])

        # Prepare message with dev guidance if available
        current_message = user_input
        if pending_dev_messages[session_id]:
            dev_msg = pending_dev_messages[session_id].pop(0)
            current_message = f"""[DEVELOPER MESSAGE - CRITICAL GUIDANCE]
{dev_msg}
[END DEVELOPER MESSAGE]
User: {user_input}"""
            print(f"[Applying supervisor guidance]")

        conversation_context.append({
            "role": "user",
            "content": current_message
        })

        if len(conversation_context) > 1:
            history_summary = "Previous conversation:\n"
            for msg in conversation_context[:-1]:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    history_summary += f"User: {content}\n"
                else:
                    history_summary += f"You (Assistant): {content}\n"

            full_message = f"{history_summary}\nCurrent user message: {current_message}"
        else:
            full_message = current_message

        if session_id not in help_agents:
            help_agents[session_id] = create_help_assistant(session_id)
        agent_with_context = help_agents[session_id]

        result = Runner.run_streamed(agent_with_context, full_message)
        assistant_response = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
                assistant_response += event.data.delta

        print()  # New line at end

        # Add assistant response to history and trigger supervisor
        conversation_context.append({
            "role": "assistant",
            "content": assistant_response
        })

        on_conversation_change(session_id, [{
            "role": "assistant",
            "content": assistant_response
        }])


if __name__ == "__main__":
    # Demo
    session_id = "demo_session_always"
    asyncio.run(run_conversation_with_always_on_supervisor(session_id))

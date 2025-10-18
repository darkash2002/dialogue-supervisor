import os
import asyncio
from typing import Optional
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, ModelSettings
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel
from openai import AsyncOpenAI

try:
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
except Exception:
    raise Exception("Failed to load OpenAI api key.")

# OpenAI client for structured outputs
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class SupervisorResponse(BaseModel):
    """Structured supervisor response."""
    intervene: bool
    developer_message: str | None  # Only if intervene=True


# Session state management
active_supervisor_threads = {}  # session_id -> asyncio.Task
conversation_histories = {}  # session_id -> list of messages
supervisor_agents = {}  # session_id -> Agent
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


async def start_supervisor_analysis(session_id: str):
    """
    Run supervisor analysis in background and store result.
    """
    try:
        dev_message = await analyze_with_supervisor(session_id)
        if dev_message:
            pending_dev_messages[session_id].append(dev_message)
            print(f"\n[Supervisor guidance ready for next turn]")
    except Exception as e:
        print(f"\n[Supervisor analysis failed: {e}]")
    finally:
        # Clear the active thread when done
        if session_id in active_supervisor_threads:
            del active_supervisor_threads[session_id]


def call_for_supervisor(session_id: str) -> bool:
    """
    Tool for assistant to request supervisor help.
    Creates supervisor thread if doesn't exist.
    Returns True on successful trigger.
    """
    if session_id in active_supervisor_threads:
        return True
    task = asyncio.create_task(start_supervisor_analysis(session_id))
    active_supervisor_threads[session_id] = task
    return True


def should_auto_trigger_supervisor(session_id: str, turn_count: int) -> bool:
    """Check if we should auto-trigger supervisor (6 turns rule)."""
    # Don't trigger if supervisor already exists
    if session_id in active_supervisor_threads:
        return False

    return turn_count >= 6


def create_help_assistant(session_id: str):
    """Create help assistant with supervisor call capability."""
    with open("prompts/help_assistant_with_supervisor.txt", "r") as f:
        instructions = f.read()

    @function_tool
    def request_supervisor_help() -> str:
        """
        Call this tool when the conversation isn't going well or when you're unsure.
        This will trigger supervisor analysis to provide you with guidance.
        """
        call_for_supervisor(session_id)
        return "Supervisor is analyzing the conversation. Guidance will be provided soon as messages."

    agent = Agent(
        name="HelpAssistant",
        instructions=instructions,
        model="gpt-4.1-nano",
        tools=[request_supervisor_help],
    )
    return agent


async def run_conversation_with_triggers(session_id: str):
    """
    Run conversation with triggered supervision.
    Supervisor analyzes after each turn in background.
    """
    conversation_histories[session_id] = []
    pending_dev_messages[session_id] = []
    turn_count = 0
    conversation_context = []

    print("Welcome to the demo. How can we help you creating a food or workout plan? Type /exit to quit.")
    while True:
        user_input = input("\n> ")
        if user_input.lower() in ["/exit"]:
            print("Goodbye!")
            break

        conversation_histories[session_id].append({
            "role": "user",
            "content": user_input
        })
        current_message = user_input
        if pending_dev_messages[session_id]:
            dev_msg = pending_dev_messages[session_id].pop(0)
            current_message = f"""[DEVELOPER MESSAGE - CRITICAL GUIDANCE]\n{dev_msg}\n[END DEVELOPER MESSAGE]
            User: {user_input}"""
            print(f"[Applying supervisor guidance]")
        conversation_context.append({
            "role": "user",
            "content": current_message
        })
        if should_auto_trigger_supervisor(session_id, turn_count):
            print("[Auto-triggering supervisor - analyzing in background]")
            call_for_supervisor(session_id)

        if session_id not in help_agents:
            help_agents[session_id] = create_help_assistant(session_id)
        agent_with_context = help_agents[session_id]
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

        result = Runner.run_streamed(agent_with_context, full_message)
        assistant_response = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
                assistant_response += event.data.delta

        print()  # New line at end

        # Add assistant response to history
        conversation_histories[session_id].append({
            "role": "assistant",
            "content": assistant_response
        })

        conversation_context.append({
            "role": "assistant",
            "content": assistant_response
        })

        # Trigger supervisor analysis after each turn (in background)
        if session_id not in active_supervisor_threads:
            call_for_supervisor(session_id)
            await asyncio.sleep(0.01)
        turn_count += 1


if __name__ == "__main__":
    # Demo
    session_id = "demo_session_2"
    asyncio.run(run_conversation_with_triggers(session_id))

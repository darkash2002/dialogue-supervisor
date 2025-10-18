# Dialogue Supervisor

An experimental system exploring AI conversation quality control through supervised learning. This project compares three distinct approaches to monitoring and improving AI assistant behavior in real-time conversations.

## Overview

Dialogue Supervisor demonstrates how a lightweight supervisory AI can monitor and steer a faster primary assistant during conversations. The system explores the trade-offs between response speed, conversation quality, and computational overhead through three different supervision modes.

### The Three Modes

| Mode | File | Supervision | Use Case |
|------|------|-------------|----------|
| **Mode 1** | `no_supervisor.py` | None | Baseline - fastest responses, no oversight |
| **Mode 2** | `supervisor_trigger.py` | On-demand | Manual triggering + auto after 6 turns |
| **Mode 3** | `supervisor_always.py` | Continuous | Real-time background monitoring |

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd dialogue-supervisor
```

2. Install dependencies:
```bash
uv pip install openai python-dotenv
```

3. Create a `.env` file with your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### Running the Modes

Each mode is a standalone script. Choose one to run:

```bash
# Mode 1: Fast baseline (no supervision)
uv run python no_supervisor.py

# Mode 2: Triggered supervision (balanced)
uv run python supervisor_trigger.py

# Mode 3: Always-on supervision (maximum oversight)
uv run python supervisor_always.py
```

## How It Works

### Primary Assistant
- **Model**: GPT-4.1-nano (fast, efficient)
- **Domain**: Meal planning and workout guidance
- **Capabilities**: Web search for up-to-date information
- **Behavior**: Responds directly to user queries with domain expertise

### Supervisor (Modes 2 & 3)
- **Model**: GPT-5-mini with minimal reasoning
- **Role**: Analyzes conversation quality and assistant behavior
- **Output**: Developer messages that guide the assistant
- **Design**: Invisible to users - steers through backend messages

## Architecture

```
User Input
    ↓
Primary Assistant (GPT-4.1-nano)
    ↓                     ↓
Response            Conversation Log
                          ↓
                    Supervisor (GPT-5-mini)
                          ↓
                  Developer Message
                          ↓
                  [Injected into context]
                          ↓
                Primary Assistant (steered)
```

## For more info - look at the substack blog post

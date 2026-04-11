# simple-tool-use

A minimal tool-use agent built directly on the Anthropic Python SDK.

## Overview

Sends a user message to Claude along with a set of tool schemas. Claude reasons about which tools to call and in what order, the agent executes them and feeds the results back, and the loop continues until Claude has enough information to return a final answer.

The script defines three tools — `get_weather`, `calculate`, and `get_float_ceil` — and prompts Claude with a question that requires all three in sequence: look up the weather in Paris, calculate 15% of the temperature, then return the ceiling of that value.

The goal isn't the task itself but to demonstrate what the loop exposes: how tool schemas are defined, how tool calls are structured in the response, how results are fed back, and how the agent knows when to stop.

## Usage

```shell
# Install env
uv sync --active

# Run the agent
ANTHROPIC_API_KEY=<your-key> python agent.py
```

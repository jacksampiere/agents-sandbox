"""
Minimal tool-use agent: raw Anthropic SDK, no frameworks.

Demonstrates the full agentic loop:
  define tools → send message → parse tool_use blocks →
  execute tools → feed results back → repeat until end_turn
"""

import anthropic
import math

client = anthropic.Anthropic()

# ── Tool implementations ───────────────────────────────────────────────────────
# Plain Python functions. Nothing special about them.
# The SDK has no idea they exist — the user decides when to call them.


def get_weather(city: str) -> str:
    """Spoof weather data."""
    data = {
        "paris": "18°C, partly cloudy",
        "boston": "24°C, sunny",
        "seattle": "12°C, rainy",
    }
    return data.get(city.lower(), f"No data for '{city}'")


def calculate(expression: str) -> str:
    """Evaluate a basic math expression. Uses eval() on digits/operators only."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: only numeric expressions are supported"
    try:
        result = eval(expression, {"__builtins__": {}})  # no builtins = safer eval
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_float_ceil(x):
    """Cast a string representation of a float to its ceiling."""
    return str(int(math.ceil(float(x))))


# ── Tool schemas ───────────────────────────────────────────────────────────────
# These are what the user sends to the API so Claude knows which tools exist.
# The "input_schema" follows JSON Schema — Claude uses it to decide which
# arguments to pass, and the API validates the structure.

TOOLS = [
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for a city. "
            "Returns temperature and conditions as a string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city, e.g. 'Paris'",
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate a basic arithmetic expression, e.g. '18 * 0.15'. "
            "Supports +, -, *, /, parentheses, and decimals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A numeric expression to evaluate, e.g. '100 / 4'",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_float_ceil",
        "description": (
            "Cast a string representation of a float to its ceiling, e.g. '3.3' to '4'. "
            "Returns the ceiling value as a string, e.g. '4'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "string",
                    "description": "A string representation of a float to be converted to its ceiling, e.g. '3.3'",
                },
            },
            "required": ["x"],
        },
    },
]

# ── Tool dispatcher ────────────────────────────────────────────────────────────
# Maps tool names to functions. This is the bridge between Claude's tool_use
# block (which contains a name + input dict) and your Python functions.

TOOL_FUNCTIONS = {
    "get_weather": lambda args: get_weather(args["city"]),
    "calculate": lambda args: calculate(args["expression"]),
    "get_float_ceil": lambda args: get_float_ceil(args["x"]),
}


def run_agent(user_message: str) -> str:
    """
    Run the agentic loop for a single user message.
    Returns the final text response from Claude.

    The loop:
      1. Send current messages + tool schemas to Claude
      2. If stop_reason == "tool_use": execute tools, append results, go to 1
      3. If stop_reason == "end_turn": extract text, return it
    """
    messages = [{"role": "user", "content": user_message}]

    print(f"\n{'=' * 60}")
    print(f"USER: {user_message}")
    print(f"{'=' * 60}")

    while True:
        # ── API call ───────────────────────────────────────────────────────────
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        print(f"\n[stop_reason: {response.stop_reason}]")

        for block in response.content:
            if block.type == "text":
                print(f"[assistant text]: {block.text}")
            elif block.type == "tool_use":
                print(f"[tool_use]: {block.name}({block.input})")

        # ── Termination check ──────────────────────────────────────────────────
        if response.stop_reason == "end_turn":
            # Extract and return the final text response
            for block in response.content:
                if block.type == "text":
                    return block.text
            return "(no text in final response)"

        # ── Tool execution ─────────────────────────────────────────────────────
        # If we're still here, stop_reason == "tool_use".
        # We need to:
        #   (a) append the assistant's message (containing tool_use blocks) to history
        #   (b) execute each tool
        #   (c) append a user message containing all the tool_result blocks
        # Both (a) and (c) must happen before the next API call.

        # (a) Append the assistant's response so the conversation history shows
        # what the model said and which tools it decided to call (including the tool_use ids)
        messages.append({"role": "assistant", "content": response.content})

        # (b) + (c) Build tool_result blocks for every tool_use block in the response
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            fn = TOOL_FUNCTIONS.get(block.name)
            if fn is None:
                result_content = f"Error: unknown tool '{block.name}'"
            else:
                result_content = fn(block.input)

            print(f"[tool_result]: {block.name} → {result_content}")

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,  # links this result to the tool call above
                    "content": result_content,
                }
            )

        # Append tool results as a user turn, referencing each tool_use_id
        # from above (the API requires both to be present before the next call)
        # Important: one user message can contain multiple tool_result blocks
        # if Claude called multiple tools in the same turn.
        messages.append({"role": "user", "content": tool_results})


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # This prompt forces Claude to all three tools:
    # it must look up weather, then calculate a percentage of the temperature, then cast to its ceiling.
    final = run_agent(
        "What's the weather in Paris right now? "
        "Then tell me what 15% of that temperature (in °C) is. "
        "Then tell me what the celing of that percentage is."
    )
    print(f"\n{'=' * 60}")
    print(f"FINAL ANSWER:\n{final}")
    print(f"{'=' * 60}\n")

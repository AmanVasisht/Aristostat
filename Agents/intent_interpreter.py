"""
FILE: agents/intent_interpreter.py
-------------------------------------
Intent Interpreter agent — wires together all components and exposes
the public run_intent_interpreter() entry point for the LangGraph orchestrator.

Imports:
  - LLM model
  - Prompt  ← prompts/intent_interpreter_prompt.py
  - Tools   ← tools/intent_interpreter_tools.py
  - Engine  ← core/intent_engine.py  (used via tools)
  - Schemas ← schemas/intent_schema.py (IntentOutput travels in LangGraph state)
"""

from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from Prompts.intent_interpreter import INTENT_INTERPRETER_SYSTEM_PROMPT
from Tools.intent_interpreter import (
    INTENT_INTERPRETER_TOOLS,
    init_intent_store,
    get_intent_store,
)


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


# ─────────────────────────────────────────────
# AGENT FACTORY
# ─────────────────────────────────────────────

def create_intent_interpreter_agent():
    """
    Create and return the Intent Interpreter ReAct agent.
    Called fresh per invocation to avoid stale state.
    """
    return create_react_agent(
        model=model,
        tools=INTENT_INTERPRETER_TOOLS,
        prompt=INTENT_INTERPRETER_SYSTEM_PROMPT
    )


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def run_intent_interpreter(
    profiler_output: dict,
    user_query: str,
) -> dict[str, Any]:
    """
    Entry point called by the LangGraph orchestrator (main.py).

    Args:
        profiler_output: The ProfilerOutput dict from the Data Profiler agent.
        user_query:      The raw query string from the user.

    Returns:
        {
          "messages":      Full LangGraph message history.
          "final_response": Human-readable interpretation summary shown to user.
          "intent_output":  Raw IntentOutput dict for downstream agents.
                            Contains methodologist_bypass flag, columns, goal, etc.
        }
    """
    # Seed the store before invoking the agent
    init_intent_store(
        profiler_output=profiler_output,
        original_query=user_query,
    )

    agent = create_intent_interpreter_agent()

    # The agent retrieves query and profiler output via tools,
    # so we just need a simple trigger message here.
    content = "Please interpret the user's analysis intent and produce a structured output."

    result = agent.invoke({"messages": [HumanMessage(content=content)]})

    # ── Extract final human-readable response ──
    final_response = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
            final_response = msg.content
            break

    # ── Retrieve validated IntentOutput from store ──
    store = get_intent_store()
    intent_output = store.get("intent_output")
    intent_output_dict = intent_output.model_dump() if intent_output else {}

    return {
        "messages": result["messages"],
        "final_response": final_response,
        "intent_output": intent_output_dict,
    }


# ─────────────────────────────────────────────
# STANDALONE TEST RUNNER
# Usage: python agents/intent_interpreter.py
# (requires a mock profiler_output — see example below)
# ─────────────────────────────────────────────

# if __name__ == "__main__":
#     # ── Mock profiler output for standalone testing ──
#     mock_profiler_output = {
#         "n_rows": 500,
#         "n_cols": 4,
#         "continuous_columns": [
#             {"column": "age",    "mean": 34.2, "std": 8.1, "skewness": 0.3,
#              "missing_pct": 0.0, "missing_severity": "low", "anomaly_count": 2},
#             {"column": "salary", "mean": 55000, "std": 12000, "skewness": 1.2,
#              "missing_pct": 0.01, "missing_severity": "low", "anomaly_count": 5},
#         ],
#         "categorical_columns": [
#             {"column": "gender",     "cardinality": 2,  "mode": "M", "mode_frequency": 0.54,
#              "missing_pct": 0.0, "missing_severity": "low", "class_imbalance_flag": False},
#             {"column": "department", "cardinality": 5,  "mode": "Engineering", "mode_frequency": 0.40,
#              "missing_pct": 0.0, "missing_severity": "low", "class_imbalance_flag": False},
#         ],
#         "warnings": [],
#         "fatal_errors": [],
#     }

#     mock_query = "I want to check if salary differs significantly between genders"

#     print("\n" + "=" * 60)
#     print("  ARISTOSTAT — Intent Interpreter Agent")
#     print("=" * 60 + "\n")
#     print(f"User query: {mock_query}\n")

#     output = run_intent_interpreter(
#         profiler_output=mock_profiler_output,
#         user_query=mock_query,
#     )

#     print(output["final_response"])
#     print("\n" + "=" * 60)
#     print("Intent output keys:", list(output["intent_output"].keys()))
#     print("Methodologist bypass:", output["intent_output"].get("methodologist_bypass"))
#     print("Confidence:", output["intent_output"].get("confidence"))
"""
FILE: agents/assumption_checker.py
------------------------------------
Assumption Checker agent — wires together all components and exposes
the public run_assumption_checker() entry point for the LangGraph orchestrator.

Imports:
  - LLM model
  - Prompt   ← prompts/assumption_checker_prompt.py
  - Tools    ← tools/assumption_checker_tools.py
  - Engine   ← core/assumption_engine.py  (called via tools)
  - Config   ← configs/assumptions.py
  - Schemas  ← schemas/assumption_checker_schema.py
"""

from typing import Any

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from Prompts.assumption_checker import ASSUMPTION_CHECKER_SYSTEM_PROMPT
from Tools.assumption_checker import (
    ASSUMPTION_CHECKER_TOOLS,
    init_assumption_store,
    get_assumption_store,
)


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
from langchain_groq import ChatGroq
model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0
)


# ─────────────────────────────────────────────
# AGENT FACTORY
# ─────────────────────────────────────────────

def create_assumption_checker_agent():
    """Create and return the Assumption Checker ReAct agent."""
    return create_react_agent(
        model=model,
        tools=ASSUMPTION_CHECKER_TOOLS,
        prompt=ASSUMPTION_CHECKER_SYSTEM_PROMPT
    )


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def run_assumption_checker(
    methodologist_output: dict,
    cleaned_df: pd.DataFrame,
    profiler_output: dict,
) -> dict[str, Any]:
    """
    Entry point called by the LangGraph orchestrator (main.py).

    Args:
        methodologist_output: MethodologistOutput dict — contains selected test + column roles.
        cleaned_df:           Cleaned pd.DataFrame from the Preprocessor agent.
        profiler_output:      ProfilerOutput dict from the Data Profiler agent.

    Returns:
        {
          "messages":          Full LangGraph message history.
          "final_response":    Human-readable assumption check summary shown to user.
          "checker_output":    Raw AssumptionCheckerOutput dict for orchestrator routing.
                               Key fields:
                                 - has_failures: bool
                                 - proceed_to_statistician: bool
                                 - pending_manual_confirmations: list[str]
                                 - results: list of individual AssumptionResult dicts
        }
    """
    init_assumption_store(
        methodologist_output=methodologist_output,
        cleaned_df=cleaned_df,
        profiler_output=profiler_output,
    )

    agent = create_assumption_checker_agent()

    content = "Please check all pre-test assumptions for the selected statistical test."

    result = agent.invoke({"messages": [HumanMessage(content=content)]})

    # ── Extract final human-readable response ──
    final_response = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
            final_response = msg.content
            break

    # ── Retrieve checker output from store ──
    store = get_assumption_store()
    checker_output = store.get("checker_output")
    checker_output_dict = checker_output.model_dump() if checker_output else {}

    return {
        "messages":       result["messages"],
        "final_response": final_response,
        "checker_output": checker_output_dict,
    }


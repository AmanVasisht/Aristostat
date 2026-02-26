from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from prompts.data_profiler_prompt import DATA_PROFILER_SYSTEM_PROMPT
from custom_proj.Tools.data_profiler import DATA_PROFILER_TOOLS, get_dataframe_store
from core.profiler_engine import profile_dataframe


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

def create_data_profiler_agent():
    """
    Create and return the Data Profiler ReAct agent.
    Called fresh for each invocation to avoid stale state.
    """
    return create_react_agent(
        model=model,
        tools=DATA_PROFILER_TOOLS,
        prompt=DATA_PROFILER_SYSTEM_PROMPT
    )


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def run_data_profiler(filepath: str, user_message: str = None) -> dict[str, Any]:
    """
    Entry point called by the LangGraph orchestrator (main.py).

    Args:
        filepath:     Path to the CSV file uploaded by the user.
        user_message: Optional analysis goal stated by the user.
                      If None, a generic profiling request is sent.

    Returns:
        {
          "messages":       Full LangGraph message history (for state passing).
          "final_response": Human-readable summary string from the agent.
          "profiler_output": Raw ProfilerOutput dict for downstream agents.
        }
    """
    agent = create_data_profiler_agent()

    content = (
        f"Please profile this dataset for me.\n"
        f"File path: {filepath}\n"
        + (f"My analysis goal: {user_message}" if user_message else "")
    )

    result = agent.invoke({"messages": [HumanMessage(content=content)]})

    # ── Extract final human-readable response ──
    final_response = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
            final_response = msg.content
            break

    # ── Build raw profiler output dict for downstream agents ──
    # Re-runs the pure engine (no LLM cost) on the already-loaded DataFrame.
    profiler_output_dict = {}
    store = get_dataframe_store()
    if "current" in store:
        try:
            profiler_output_dict = profile_dataframe(store["current"]).model_dump()
        except Exception:
            pass

    return {
        "messages": result["messages"],
        "final_response": final_response,
        "profiler_output": profiler_output_dict,
    }
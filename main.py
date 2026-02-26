"""
FILE: main.py
--------------
LangGraph orchestrator for Aristostat.
Wires all nine agents into a StateGraph with conditional edges
and human-in-the-loop interrupt() checkpoints.

Pipeline flow:
  data_profiler
      ↓ [interrupt — show stats, confirm proceed]
  intent_interpreter
      ↓ [interrupt — confirm interpretation]
  methodologist
      ↓ [interrupt — confirm test selection]
  preprocessor
      ↓ [interrupt — show cleaning summary]
  assumption_checker
      ↓ [interrupt — show pass/fail, rectify or proceed?]
      ↓ if failures → rectification_strategist (pre_test)
          ↓ if test_switch → back to assumption_checker (loop, max 3)
          ↓ if proceed/correction → statistician
      ↓ if all pass → statistician
  statistician
      ↓
  model_critic
      ↓ [interrupt if failures — rectify or proceed?]
      ↓ if failures → rectification_strategist (post_test) → statistician
      ↓ if pass/skipped → final_report
  final_report
      ↓ [END]

Human-in-the-loop:
  LangGraph interrupt() pauses execution and surfaces a value to the caller.
  The caller collects user input and resumes via graph.invoke() with
  Command(resume=user_response).

State:
  AristostatState TypedDict — all pipeline outputs stored as optional fields.
  rectification_attempt in state — incremented by orchestrator at each loop.
"""

import pandas as pd
from typing import TypedDict, Annotated, Any
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

# ── Agent entry points ──
from agents.data_profiler       import run_data_profiler
from agents.intent_interpreter  import run_intent_interpreter
from agents.methodologist       import run_methodologist
from agents.preprocessor        import run_preprocessor
from agents.assumption_checker  import run_assumption_checker
from agents.rectification_strategist import run_rectification_strategist
from agents.statistician        import run_statistician
from agents.model_critic        import run_model_critic
from agents.final_report        import run_final_report

# ── Fitted model accessor (passed in memory to Model Critic) ──
from tools.statistician_tools   import get_fitted_model
import time

# ─────────────────────────────────────────────
# STATE SCHEMA
# TypedDict — all fields optional, populated as pipeline progresses
# ─────────────────────────────────────────────

class AristostatState(TypedDict, total=False):
    # ── Inputs ──
    csv_path:             str
    user_query:           str

    # ── Working data ──
    raw_df:               Any             # pd.DataFrame
    cleaned_df:           Any             # pd.DataFrame (after preprocessing)
    rectified_df:         Any             # pd.DataFrame (after rectification transform)

    # ── Agent outputs ──
    profiler_output:      dict
    intent_output:        dict
    methodologist_output: dict
    preprocessor_output:  dict
    checker_output:       dict
    rectification_output: dict | None
    statistician_output:  dict
    critic_output:        dict | None
    report_output:        dict

    # ── Rectification loop counter (lives in state, not in RectificationOutput) ──
    rectification_attempt: int            # incremented by orchestrator each loop
    rectification_phase:   str            # "pre_test" | "post_test"

    # ── Human checkpoint responses ──
    user_confirmed_proceed:    bool       # generic proceed confirmation
    user_rectify_or_proceed:   str        # "rectify" | "proceed"
    user_chosen_solution_id:   str | None # solution chosen in rectification

    # ── Routing flags ──
    fatal_error:          str | None      # set if pipeline must stop
    bypass_assumption_checker: bool       # True if rectification switched test


# ─────────────────────────────────────────────
# NODE FUNCTIONS
# Each node runs one agent and updates state.
# ─────────────────────────────────────────────

def node_data_profiler(state: AristostatState) -> AristostatState:
    """Loads CSV, profiles dataset, shows stats to user."""
    result = run_data_profiler(csv_path=state["csv_path"])

    # ── Human checkpoint: show profiler summary, ask to proceed ──
    user_response = interrupt({
        "message": result["final_response"],
        "prompt":  "Shall I proceed with the analysis? (yes/no)",
        "type":    "confirm",
    })

    if str(user_response).strip().lower() not in ("yes", "y"):
        return {**state, "fatal_error": "User stopped pipeline after data profiling."}

    return {
        **state,
        "raw_df":           result["raw_df"],
        "profiler_output":  result["profiler_output"],
    }


def node_intent_interpreter(state: AristostatState) -> AristostatState:
    """Parses user query into structured intent."""
    result = run_intent_interpreter(
        user_query=state["user_query"],
        profiler_output=state["profiler_output"],
    )

    # ── Fatal column error ──
    intent = result.get("intent_output", {})
    if intent.get("fatal_error"):
        fatal = interrupt({
            "message": result["final_response"],
            "prompt":  "Please correct the column name and re-submit your query.",
            "type":    "fatal_error",
        })
        return {**state, "fatal_error": result["final_response"]}

    # ── Open-ended: user picks a suggested combination ──
    if intent.get("intent_type") == "open_ended" and intent.get("suggested_combinations"):
        combo_response = interrupt({
            "message": result["final_response"],
            "prompt":  "Please choose one of the suggested combinations (enter the number).",
            "type":    "choose_combination",
            "options": intent["suggested_combinations"],
        })
        # Re-run intent interpreter with chosen combination
        chosen_query = _resolve_combination_choice(
            combo_response, intent["suggested_combinations"]
        )
        result = run_intent_interpreter(
            user_query=chosen_query,
            profiler_output=state["profiler_output"],
        )

    # ── Confirm interpretation ──
    user_response = interrupt({
        "message": result["final_response"],
        "prompt":  "Does this correctly capture your intent? (yes/no, or describe correction)",
        "type":    "confirm",
    })

    # If user corrects — re-run with their correction
    if str(user_response).strip().lower() not in ("yes", "y"):
        result = run_intent_interpreter(
            user_query=str(user_response),
            profiler_output=state["profiler_output"],
        )

    return {**state, "intent_output": result["intent_output"]}


def node_methodologist(state: AristostatState) -> AristostatState:
    """Selects the appropriate statistical test."""
    time.sleep(2)
    result = run_methodologist(
        intent_output=state["intent_output"],
        profiler_output=state["profiler_output"],
    )

    # ── Human checkpoint: show test selection + reasoning ──
    user_response = interrupt({
        "message": result["final_response"],
        "prompt":  "Shall I proceed with this test? (yes/no)",
        "type":    "confirm",
    })

    if str(user_response).strip().lower() not in ("yes", "y"):
        return {**state, "fatal_error": "User stopped pipeline after test selection."}

    return {**state, "methodologist_output": result["methodologist_output"]}


def node_preprocessor(state: AristostatState) -> AristostatState:
    """Cleans the raw dataset."""
    time.sleep(2)
    result = run_preprocessor(
        raw_df=state["raw_df"],
        profiler_output=state["profiler_output"],
    )

    # ── Fatal: high missingness ──
    if result["preprocessor_output"].get("fatal_error"):
        interrupt({
            "message": result["final_response"],
            "type":    "fatal_error",
        })
        return {**state, "fatal_error": result["preprocessor_output"]["fatal_error"]}

    # ── Human checkpoint: show cleaning summary ──
    user_response = interrupt({
        "message": result["final_response"],
        "prompt":  "Shall I proceed with the cleaned data? (yes/no)",
        "type":    "confirm",
    })

    if str(user_response).strip().lower() not in ("yes", "y"):
        return {**state, "fatal_error": "User stopped pipeline after preprocessing."}

    return {
        **state,
        "cleaned_df":          result["cleaned_df"],
        "preprocessor_output": result["preprocessor_output"],
    }


def node_assumption_checker(state: AristostatState) -> AristostatState:
    """Runs pre-test assumption checks."""
    result = run_assumption_checker(
        methodologist_output=state["methodologist_output"],
        cleaned_df=state.get("rectified_df") or state["cleaned_df"],
        profiler_output=state["profiler_output"],
    )

    checker_output = result["checker_output"]

    # ── All pass — no checkpoint needed, proceed silently ──
    if not checker_output.get("has_failures"):
        # Still show results but auto-proceed
        interrupt({
            "message": result["final_response"],
            "prompt":  "All assumptions met. Proceeding to run the test.",
            "type":    "info",
        })
        return {
            **state,
            "checker_output":           checker_output,
            "user_rectify_or_proceed":  "proceed",
        }

    # ── Failures found — ask user to rectify or proceed ──
    user_response = interrupt({
        "message": result["final_response"],
        "prompt":  "Would you like to rectify the failed assumptions or proceed anyway? (rectify/proceed)",
        "type":    "rectify_or_proceed",
    })

    decision = str(user_response).strip().lower()

    return {
        **state,
        "checker_output":          checker_output,
        "user_rectify_or_proceed": "rectify" if "rectify" in decision else "proceed",
    }


def node_rectification_strategist(state: AristostatState) -> AristostatState:
    """Proposes and applies rectification solutions."""
    attempt = state.get("rectification_attempt", 1)
    phase   = state.get("rectification_phase", "pre_test")
    df      = state.get("rectified_df") or state["cleaned_df"]

    # ── Get failed assumptions ──
    checker = state.get("checker_output", {})
    critic  = state.get("critic_output", {})

    if phase == "pre_test":
        failed = [
            r["name"] for r in checker.get("results", [])
            if r.get("status") == "failed"
        ]
    else:
        failed = [
            r["name"] for r in critic.get("results", [])
            if r.get("status") == "failed"
        ]

    result = run_rectification_strategist(
        failed_assumptions=failed,
        phase=phase,
        cleaned_df=df,
        methodologist_output=state["methodologist_output"],
        rectification_attempt=attempt,
        max_attempts=3,
    )

    rect_output = result["rectification_output"]

    # ── Human checkpoint: show solutions, user picks one ──
    user_response = interrupt({
        "message": result["final_response"],
        "prompt":  "Please choose an option (enter the number, or type 'proceed' to continue despite failures).",
        "type":    "choose_solution",
        "options": rect_output.get("proposed_solutions", []),
    })

    # ── User chose to proceed despite failure ──
    if str(user_response).strip().lower() == "proceed":
        final_result = run_rectification_strategist(
            failed_assumptions=failed,
            phase=phase,
            cleaned_df=df,
            methodologist_output=state["methodologist_output"],
            rectification_attempt=attempt,
            max_attempts=3,
        )
        # Re-invoke with accept_violation
        from tools.rectification_tools import (
            init_rectification_store, get_rectification_store,
            accept_violation_and_proceed,
        )
        init_rectification_store(
            failed_assumptions=failed,
            phase=phase,
            cleaned_df=df,
            methodologist_output=state["methodologist_output"],
            rectification_attempt=attempt,
        )
        accept_violation_and_proceed.invoke({"violation_names": ",".join(failed)})
        rect_output = get_rectification_store()["rectification_output"].model_dump()

        return {
            **state,
            "rectification_output":    rect_output,
            "rectified_df":            df,
            "user_rectify_or_proceed": "proceed",
        }

    # ── User chose a solution — apply it ──
    solution_id = _resolve_solution_choice(user_response, rect_output.get("proposed_solutions", []))

    from tools.rectification_tools import (
        init_rectification_store, get_rectification_store,
    )
    from core.rectification_engine import build_rectification_output
    from schemas.rectification_schema import RectificationPhase

    ph = RectificationPhase(phase)
    rectified_df, applied_output = build_rectification_output(
        failed_assumptions=failed,
        phase=ph,
        chosen_solution_id=solution_id,
        df=df,
        dependent_var=state["methodologist_output"].get("dependent_variable"),
        independent_vars=state["methodologist_output"].get("independent_variables", []),
        rectification_attempt=attempt,
        max_attempts=3,
    )

    rect_output = applied_output.model_dump()

    # ── If test was switched, update methodologist output ──
    updated_methodologist = state["methodologist_output"]
    if applied_output.new_test:
        updated_methodologist = {
            **state["methodologist_output"],
            "selected_test": applied_output.new_test,
        }

    return {
        **state,
        "rectification_output":    rect_output,
        "rectified_df":            rectified_df,
        "methodologist_output":    updated_methodologist,
        "rectification_attempt":   attempt + 1,
        "user_rectify_or_proceed": "rectified",
    }


def node_statistician(state: AristostatState) -> AristostatState:
    """Executes the statistical test."""
    time.sleep(3)
    df = state.get("rectified_df") or state["cleaned_df"]

    result = run_statistician(
        methodologist_output=state["methodologist_output"],
        cleaned_df=df,
        rectification_output=state.get("rectification_output"),
    )

    return {
        **state,
        "statistician_output": result["statistician_output"],
    }


def node_model_critic(state: AristostatState) -> AristostatState:
    """Runs post-test model checks (regression only)."""
    time.sleep(3)
    df = state.get("rectified_df") or state["cleaned_df"]

    result = run_model_critic(
        statistician_output=state["statistician_output"],
        fitted_model=get_fitted_model(),    # retrieved from statistician_tools memory
        cleaned_df=df,
        methodologist_output=state["methodologist_output"],
    )

    critic_output = result["critic_output"]

    # ── Non-regression — no checks, proceed silently ──
    if not critic_output.get("checks_applicable", False):
        return {**state, "critic_output": critic_output}

    # ── All pass — show results, proceed ──
    if not critic_output.get("has_failures"):
        interrupt({
            "message": result["final_response"],
            "prompt":  "All post-test checks passed. Proceeding to final report.",
            "type":    "info",
        })
        return {
            **state,
            "critic_output":           critic_output,
            "user_rectify_or_proceed": "proceed",
        }

    # ── Failures found — ask user ──
    user_response = interrupt({
        "message": result["final_response"],
        "prompt":  "Would you like to rectify the post-test failures or proceed anyway? (rectify/proceed)",
        "type":    "rectify_or_proceed",
    })

    decision = str(user_response).strip().lower()

    return {
        **state,
        "critic_output":           critic_output,
        "rectification_phase":     "post_test",
        "user_rectify_or_proceed": "rectify" if "rectify" in decision else "proceed",
    }


def node_final_report(state: AristostatState) -> AristostatState:
    """Generates the final report in chat and as a .docx file."""
    time.sleep(3)
    result = run_final_report(
        original_query=state["user_query"],
        profiler_output=state["profiler_output"],
        preprocessor_output=state["preprocessor_output"],
        methodologist_output=state["methodologist_output"],
        checker_output=state["checker_output"],
        statistician_output=state["statistician_output"],
        rectification_output=state.get("rectification_output"),
        critic_output=state.get("critic_output"),
    )

    return {**state, "report_output": result["report_output"]}


# ─────────────────────────────────────────────
# CONDITIONAL EDGE FUNCTIONS
# ─────────────────────────────────────────────

def route_after_profiler(state: AristostatState) -> str:
    if state.get("fatal_error"):
        return END
    return "intent_interpreter"


def route_after_intent(state: AristostatState) -> str:
    if state.get("fatal_error"):
        return END
    return "methodologist"


def route_after_methodologist(state: AristostatState) -> str:
    if state.get("fatal_error"):
        return END
    return "preprocessor"


def route_after_preprocessor(state: AristostatState) -> str:
    if state.get("fatal_error"):
        return END
    return "assumption_checker"


def route_after_assumption_checker(state: AristostatState) -> str:
    """
    Routes after assumption checking:
      - No failures        → statistician
      - User chose proceed → statistician
      - User chose rectify → rectification_strategist
    """
    checker = state.get("checker_output", {})
    decision = state.get("user_rectify_or_proceed", "proceed")

    if not checker.get("has_failures"):
        return "statistician"
    if decision == "rectify":
        return "rectification_strategist"
    return "statistician"


def route_after_rectification(state: AristostatState) -> str:
    """
    Routes after rectification:
      - User accepted violation → statistician
      - Test was switched       → assumption_checker (re-check new test)
      - Transform applied       → assumption_checker (re-check with new data)
      - Correction noted        → statistician (correction applied at run time)
      - Max attempts reached    → statistician
    """
    rect = state.get("rectification_output", {})
    attempt = state.get("rectification_attempt", 1)

    # Max attempts exceeded — force to statistician
    if attempt > 3:
        return "statistician"

    # User accepted violation
    if rect.get("user_accepted_violation"):
        return "statistician"

    next_step = rect.get("next_step", "assumption_checker")

    if next_step == "statistician":
        return "statistician"

    # Check if this is a post_test rectification — go back to statistician
    if state.get("rectification_phase") == "post_test":
        return "statistician"

    return "assumption_checker"


def route_after_statistician(state: AristostatState) -> str:
    return "model_critic"


def route_after_model_critic(state: AristostatState) -> str:
    """
    Routes after model critic:
      - Non-regression or no failures → final_report
      - User chose proceed             → final_report
      - User chose rectify             → rectification_strategist (post_test)
    """
    critic   = state.get("critic_output", {})
    decision = state.get("user_rectify_or_proceed", "proceed")

    if not critic.get("checks_applicable", False):
        return "final_report"
    if not critic.get("has_failures"):
        return "final_report"
    if decision == "rectify":
        return "rectification_strategist"
    return "final_report"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _resolve_combination_choice(user_response: Any, combinations: list) -> str:
    """Converts user's combination choice (number or description) to a query string."""
    try:
        idx = int(str(user_response).strip()) - 1
        combo = combinations[idx]
        cols  = combo.get("columns", [])
        goal  = combo.get("suggested_goal", "")
        return f"Please run {goal} analysis on {' and '.join(cols)}"
    except (ValueError, IndexError):
        return str(user_response)


def _resolve_solution_choice(user_response: Any, solutions: list) -> str | None:
    """Converts user's solution choice (number or solution_id) to a solution_id."""
    response = str(user_response).strip()
    try:
        idx = int(response) - 1
        if 0 <= idx < len(solutions):
            sol = solutions[idx]
            return sol.get("solution_id") if isinstance(sol, dict) else sol.solution_id
    except ValueError:
        # User typed a solution_id directly
        return response
    return None


# ─────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Builds and compiles the Aristostat LangGraph pipeline."""
    builder = StateGraph(AristostatState)

    # ── Register nodes ──
    builder.add_node("data_profiler",            node_data_profiler)
    builder.add_node("intent_interpreter",       node_intent_interpreter)
    builder.add_node("methodologist",            node_methodologist)
    builder.add_node("preprocessor",             node_preprocessor)
    builder.add_node("assumption_checker",       node_assumption_checker)
    builder.add_node("rectification_strategist", node_rectification_strategist)
    builder.add_node("statistician",             node_statistician)
    builder.add_node("model_critic",             node_model_critic)
    builder.add_node("final_report",             node_final_report)

    # ── Entry point ──
    builder.set_entry_point("data_profiler")

    # ── Conditional edges ──
    builder.add_conditional_edges("data_profiler",            route_after_profiler)
    builder.add_conditional_edges("intent_interpreter",       route_after_intent)
    builder.add_conditional_edges("methodologist",            route_after_methodologist)
    builder.add_conditional_edges("preprocessor",             route_after_preprocessor)
    builder.add_conditional_edges("assumption_checker",       route_after_assumption_checker)
    builder.add_conditional_edges("rectification_strategist", route_after_rectification)
    builder.add_conditional_edges("statistician",             route_after_statistician)
    builder.add_conditional_edges("model_critic",             route_after_model_critic)

    # ── Terminal edge ──
    builder.add_edge("final_report", END)

    # ── Compile with memory checkpointer for interrupt/resume ──
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)



def save_graph_image(output_path: str = "aristostat_graph.png") -> None:
    """
    Saves a visual diagram of the Aristostat pipeline graph as a PNG.
    Uses LangGraph's built-in Mermaid rendering via draw_mermaid_png().
    Call this once after the graph is compiled.

    Requires: pip install langgraph[draw]
    (installs playwright + chromium for Mermaid rendering)
    """
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        print(f"Graph diagram saved to: {output_path}")
    except Exception as e:
        # Fallback — print Mermaid source so user can render it manually
        print(f"Could not render PNG ({e}). Mermaid source:\n")
        print(graph.get_graph().draw_mermaid())
# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

# Module-level compiled graph — reused across invocations
graph = build_graph()
save_graph_image("aristostat_graph.png")

def run_aristostat(
    csv_path: str,
    user_query: str,
    thread_id: str = "default",
) -> dict[str, Any]:
    """
    Main entry point for running the full Aristostat pipeline.

    Args:
        csv_path:   Path to the user's CSV file.
        user_query: The user's analysis request.
        thread_id:  LangGraph thread ID for checkpointing (one per session).

    Returns:
        Final state dict after pipeline completes.
        Key fields:
          - report_output: FinalReportOutput dict (markdown + docx path)
          - fatal_error:   Set if pipeline stopped early

    Usage:
        result = run_aristostat("data.csv", "Does salary differ between genders?")

    Human-in-the-loop:
        The graph pauses at interrupt() checkpoints and raises GraphInterrupt.
        Catch it, show the message to the user, collect their response,
        then call resume_aristostat() with their response.
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: AristostatState = {
        "csv_path":              csv_path,
        "user_query":            user_query,
        "rectification_attempt": 1,
        "rectification_phase":   "pre_test",
        "fatal_error":           None,
        "rectification_output":  None,
        "critic_output":         None,
        "rectified_df":          None,
        "user_rectify_or_proceed": "proceed",
    }

    return graph.invoke(initial_state, config=config)


def resume_aristostat(
    user_response: Any,
    thread_id: str = "default",
) -> dict[str, Any]:
    """
    Resumes a paused pipeline after a human-in-the-loop interrupt.

    Args:
        user_response: The user's response to the interrupt prompt.
                       Could be "yes", "rectify", a solution number, etc.
        thread_id:     Must match the thread_id used in run_aristostat().

    Returns:
        Updated state dict. If another interrupt is hit, the graph
        pauses again and the caller must call resume_aristostat() again.
    """
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(Command(resume=user_response), config=config)


# ─────────────────────────────────────────────
# CLI RUNNER
# Usage: python main.py data.csv "Does salary differ between genders?"
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from langgraph.errors import GraphInterrupt

    if len(sys.argv) < 3:
        print("Usage: python main.py <csv_path> <query>")
        sys.exit(1)

    csv_path   = sys.argv[1]
    user_query = sys.argv[2]
    thread_id  = "cli_session"

    print("\n" + "=" * 60)
    print("  ARISTOSTAT")
    print("=" * 60)

    # ── Initial invocation ──
    try:
        state = run_aristostat(csv_path, user_query, thread_id)
        print("\nPipeline complete.")

    except GraphInterrupt as interrupt_event:
        # ── Handle interrupt loop ──
        interrupt_value = interrupt_event.args[0] if interrupt_event.args else {}

        while True:
            print(f"\n{interrupt_value.get('message', '')}")

            prompt = interrupt_value.get("prompt", "Your response: ")
            interrupt_type = interrupt_value.get("type", "confirm")

            # Show options if available
            if interrupt_type == "choose_combination" and interrupt_value.get("options"):
                for i, opt in enumerate(interrupt_value["options"], 1):
                    cols = opt.get("columns", []) if isinstance(opt, dict) else []
                    goal = opt.get("suggested_goal", "") if isinstance(opt, dict) else ""
                    print(f"  {i}. {goal} — {', '.join(cols)}")

            if interrupt_type == "choose_solution" and interrupt_value.get("options"):
                for i, opt in enumerate(interrupt_value["options"], 1):
                    desc = opt.get("description", "") if isinstance(opt, dict) else str(opt)
                    print(f"  {i}. {desc}")

            # Fatal error — no recovery
            if interrupt_type == "fatal_error":
                print("\nFatal error — pipeline stopped.")
                break

            # Info interrupt — auto-continue
            if interrupt_type == "info":
                user_input = "yes"
            else:
                user_input = input(f"\n{prompt} ").strip()

            try:
                state = resume_aristostat(user_input, thread_id)
                print("\nPipeline complete.")
                break
            except GraphInterrupt as next_interrupt:
                interrupt_value = next_interrupt.args[0] if next_interrupt.args else {}

        # ── Show final report path if generated ──
        if isinstance(state, dict):
            report = state.get("report_output", {})
            if report.get("docx_generated"):
                print(f"\nReport saved to: {report.get('docx_path')}")
            if state.get("fatal_error"):
                print(f"\nStopped: {state['fatal_error']}")
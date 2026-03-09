"""
FILE: Agents/sql_qna.py
------------------------
Entry point for the SQL QnA Agent.

Responsible for:
    - Initialising the tool store
    - Running the ReAct loop (LLM + tool calls)
    - Building and returning SqlQnAOutput

Pipeline position:
    Called from main.py → node_sql_qna_run()
    Confirm / interrupt lives in node_sql_qna_confirm() in main.py

ReAct loop flow:
    get_schema()       — always first, inspect columns and types
         ↓
    execute_sql()      — write and run DuckDB SQL
         ↓ (on error)
    fix_and_retry()    — self-correct once, re-execute
         ↓
    LLM final answer   — plain English explanation of results

Max iterations: 6
    Enough for: get_schema + execute + optional retry + explanation
    Guards against runaway tool-calling loops.
"""

from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage

from Schemas.sql_qna import SqlQnAOutput
from Tools.sql_qna import (
    init_sql_store,
    get_sql_store,
    get_schema,
    execute_sql,
    fix_and_retry,
)
from Prompts.sql_qna import SQL_QNA_SYSTEM_PROMPT


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

_MAX_ITERATIONS = 6


# ─────────────────────────────────────────────
# AGENT ENTRY POINT
# ─────────────────────────────────────────────

def run_sql_qna_agent(
    csv_path:      str,
    user_question: str,
) -> dict[str, Any]:
    """
    Entry point for the SQL QnA agent.
    Called from main.py node_sql_qna_run().

    Args:
        csv_path:      Absolute path to the uploaded CSV file.
        user_question: The user's plain English data question.

    Returns:
        {
            "sql_output":     SqlQnAOutput  — structured result
            "final_response": str           — plain English answer for display
        }
    """
    print(f"\n[SQL QnA Agent] Starting...")
    print(f"[SQL QnA Agent] question : {user_question}")
    print(f"[SQL QnA Agent] csv_path : {csv_path}")

    # ── Initialise store ──
    init_sql_store(csv_path=csv_path, user_question=user_question)

    # ── LLM + tools ──
    llm   = ChatGroq(model="qwen/qwen3-32b", temperature=0)
    tools = [get_schema, execute_sql, fix_and_retry]

    llm_with_tools = llm.bind_tools(tools)

    tool_registry = {
        "get_schema":    get_schema,
        "execute_sql":   execute_sql,
        "fix_and_retry": fix_and_retry,
    }

    # ── Conversation history for ReAct loop ──
    messages = [
        {"role": "system", "content": SQL_QNA_SYSTEM_PROMPT},
        {"role": "user",   "content": user_question},
    ]

    # ── ReAct loop ──
    final_text = ""
    iteration  = 0

    while iteration < _MAX_ITERATIONS:
        iteration += 1
        print(f"[SQL QnA Agent] iteration {iteration}")

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # ── No tool calls → agent has its final answer ──
        if not response.tool_calls:
            final_text = response.content
            print(f"[SQL QnA Agent] final answer received at iteration {iteration}")
            break

        # ── Execute each tool call and append result ──
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id   = tool_call["id"]

            print(f"[SQL QnA Agent] tool call : {tool_name}")
            print(f"[SQL QnA Agent] tool args : {tool_args}")

            if tool_name in tool_registry:
                try:
                    tool_result = tool_registry[tool_name].invoke(tool_args)
                except Exception as e:
                    tool_result = f"Tool execution error: {str(e)}"
            else:
                tool_result = f"Unknown tool requested: {tool_name}"

            print(f"[SQL QnA Agent] tool result preview: {str(tool_result)[:200]}")

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id,
                )
            )

    # ── Loop exhausted without final answer ──
    if not final_text:
        final_text = (
            "I was unable to answer your question with the available data. "
            "Please try rephrasing your question."
        )
        print(f"[SQL QnA Agent] WARNING: loop exhausted without final answer")

    # ── Read store results ──
    store = get_sql_store()

    # ── Build output schema ──
    sql_output = SqlQnAOutput(
        query_understood = user_question,
        generated_sql    = store.get("generated_sql", ""),
        result_table     = store.get("result_table", []),
        row_count        = store.get("row_count", 0),
        explanation      = final_text,
        fatal_error      = store.get("fatal_error"),
        filtered_df      = None,  # future planner integration — Use Case 2
    )

    print(f"[SQL QnA Agent] row_count   : {sql_output.row_count}")
    print(f"[SQL QnA Agent] fatal_error : {sql_output.fatal_error}")

    return {
        "sql_output":     sql_output,
        "final_response": final_text,
    }
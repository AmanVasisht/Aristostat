"""
FILE: tools/final_report_tools.py
------------------------------------
LangChain tools exposed to the Final Report ReAct agent.
The agent assembles the report, writes the interpretation,
and triggers docx generation via Node.js.
"""

import json
import os
import subprocess
import tempfile
import pandas as pd
from langchain_core.tools import tool

from core.final_report_engine import assemble_report
from Schemas.final_report import FinalReportOutput


# ─────────────────────────────────────────────
# SESSION STORE
# ─────────────────────────────────────────────

_report_store: dict = {
    "original_query":        None,
    "profiler_output":       None,
    "preprocessor_output":   None,
    "methodologist_output":  None,
    "checker_output":        None,
    "statistician_output":   None,
    "rectification_output":  None,
    "critic_output":         None,
    "report_output":         None,
}


def init_report_store(
    original_query: str,
    profiler_output: dict,
    preprocessor_output: dict,
    methodologist_output: dict,
    checker_output: dict,
    statistician_output: dict,
    rectification_output: dict | None = None,
    critic_output: dict | None = None,
) -> None:
    """Called by run_final_report() before invoking the agent."""
    _report_store["original_query"]       = original_query
    _report_store["profiler_output"]      = profiler_output
    _report_store["preprocessor_output"]  = preprocessor_output
    _report_store["methodologist_output"] = methodologist_output
    _report_store["checker_output"]       = checker_output
    _report_store["statistician_output"]  = statistician_output
    _report_store["rectification_output"] = rectification_output
    _report_store["critic_output"]        = critic_output
    _report_store["report_output"]        = None


def get_report_store() -> dict:
    return _report_store


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

@tool
def get_pipeline_summary() -> str:
    """
    Returns a structured summary of everything that happened in the pipeline —
    dataset info, test selected, assumptions checked, rectifications applied,
    test results, and post-test checks.
    Use this to understand what to cover in the report before writing it.
    """
    store = _report_store
    stat  = store.get("statistician_output", {})
    meth  = store.get("methodologist_output", {})
    rect  = store.get("rectification_output") or {}
    check = store.get("checker_output", {})
    crit  = store.get("critic_output") or {}
    prof  = store.get("profiler_output", {})

    family = stat.get("test_family", "")

    # Extract the most relevant result fields based on family
    result_summary: dict = {}
    if family == "inference" and stat.get("inference_result"):
        r = stat["inference_result"]
        result_summary = {
            "statistic":    f"{r.get('statistic_label')}={r.get('statistic')}",
            "p_value":      r.get("p_value"),
            "verdict":      r.get("verdict"),
            "effect_size":  f"{r.get('effect_size_label')}={r.get('effect_size')}",
            "group_stats":  r.get("group_stats", {}),
            "interpretation": r.get("interpretation"),
        }
    elif family == "regression" and stat.get("regression_result"):
        r = stat["regression_result"]
        result_summary = {
            "r_squared":     r.get("r_squared"),
            "adj_r_squared": r.get("adj_r_squared"),
            "f_statistic":   r.get("f_statistic"),
            "f_p_value":     r.get("f_p_value"),
            "coefficients":  [
                {"variable": c.get("variable"), "estimate": c.get("estimate"),
                 "p_value": c.get("p_value")}
                for c in r.get("coefficients", [])
            ],
            "interpretation": r.get("interpretation"),
        }
    elif family == "correlation" and stat.get("correlation_result"):
        r = stat["correlation_result"]
        result_summary = {
            "statistic":   f"{r.get('statistic_label')}={r.get('statistic')}",
            "p_value":     r.get("p_value"),
            "verdict":     r.get("verdict"),
            "strength":    r.get("correlation_strength"),
            "interpretation": r.get("interpretation"),
        }
    elif family == "dimensionality" and stat.get("dimensionality_result"):
        r = stat["dimensionality_result"]
        result_summary = {
            "n_components_selected":    r.get("n_components_selected"),
            "total_variance_explained": r.get("total_variance_explained"),
            "interpretation":           r.get("interpretation"),
        }

    return json.dumps({
        "original_query":      store.get("original_query"),
        "test_name":           stat.get("test_name"),
        "test_family":         family,
        "selection_mode":      meth.get("selection_mode"),
        "dependent_variable":  meth.get("dependent_variable"),
        "independent_variables": meth.get("independent_variables", []),
        "grouping_variable":   meth.get("grouping_variable"),
        "n_rows_original":     prof.get("n_rows"),
        "assumptions_passed":  check.get("passed_count"),
        "assumptions_failed":  check.get("failed_count"),
        "rectification_applied": bool(rect.get("chosen_solution_id")),
        "correction_applied":  stat.get("correction_applied"),
        "post_test_applicable": crit.get("checks_applicable", False),
        "post_test_failures":  crit.get("failed_count", 0),
        "result":              result_summary,
    }, indent=2)


@tool
def build_and_render_report(interpretation: str) -> str:
    """
    Assembles the full report using all pipeline outputs.
    The LLM provides the interpretation argument — a 2-4 sentence plain English
    explanation of what the results mean for the user's original question.
    This is the only part the LLM writes; everything else is assembled from data.

    Returns the markdown report as a string for display in chat.

    Args:
        interpretation: Plain English explanation of what the results mean.
                        Should directly answer the user's original question.
                        Example: "Experience is a significant positive predictor of salary.
                        For each additional year of experience, salary increases by
                        approximately £1,250 on average."
    """
    store = _report_store

    try:
        report = assemble_report(
            original_query=store.get("original_query", ""),
            profiler_output=store.get("profiler_output", {}),
            preprocessor_output=store.get("preprocessor_output", {}),
            methodologist_output=store.get("methodologist_output", {}),
            checker_output=store.get("checker_output", {}),
            statistician_output=store.get("statistician_output", {}),
            rectification_output=store.get("rectification_output"),
            critic_output=store.get("critic_output"),
            interpretation=interpretation,
        )
        _report_store["report_output"] = report
        return report.markdown_report
    except Exception as e:
        return f"ERROR: Could not assemble report — {str(e)}"


@tool
def generate_docx_report() -> str:
    """
    Generates a downloadable .docx Word document from the assembled report.
    Call this AFTER build_and_render_report has been called successfully.
    """
    report: FinalReportOutput | None = _report_store.get("report_output")
    if report is None:
        return "ERROR: No report assembled. Call build_and_render_report first."

    try:
        import os
        from docx import Document
        from docx.shared import Pt, RGBColor, Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        doc = Document()

        # ── Page margins ──
        for section in doc.sections:
            section.top_margin    = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin   = Inches(1)
            section.right_margin  = Inches(1)

        # ── Title ──
        title = doc.add_heading(report.title, level=1)
        title.runs[0].font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

        # ── Query ──
        p = doc.add_paragraph()
        p.add_run("Query: ").bold = True
        p.add_run(report.original_query)
        doc.add_paragraph()

        # ── Dataset & Preparation ──
        doc.add_heading("Dataset & Preparation", level=2)
        doc.add_paragraph(report.dataset_summary)
        doc.add_paragraph()

        # ── Test Selection ──
        doc.add_heading("Test Selection", level=2)
        doc.add_paragraph(report.test_selected)
        doc.add_paragraph()

        # ── Assumption Checks ──
        doc.add_heading("Assumption Checks", level=2)
        doc.add_paragraph(report.assumptions_summary)
        doc.add_paragraph()

        # ── Rectifications (if any) ──
        if report.rectifications_applied:
            doc.add_heading("Rectifications Applied", level=2)
            for r in report.rectifications_applied:
                doc.add_paragraph(r, style="List Bullet")
            doc.add_paragraph()

        # ── Post-test checks (if any) ──
        if report.post_test_summary:
            doc.add_heading("Post-Test Model Checks", level=2)
            doc.add_paragraph(report.post_test_summary)
            doc.add_paragraph()

        # ── Results ──
        doc.add_heading("Results", level=2)
        p = doc.add_paragraph()
        run = p.add_run(report.key_statistic)
        run.bold = True
        doc.add_paragraph(report.verdict)
        if report.effect_size:
            p = doc.add_paragraph()
            p.add_run("Effect size: ").bold = True
            p.add_run(report.effect_size)
        doc.add_paragraph()

        # ── Interpretation ──
        doc.add_heading("Interpretation", level=2)
        doc.add_paragraph(report.interpretation)
        doc.add_paragraph()

        # ── Caveats (if any) ──
        if report.caveats:
            doc.add_heading("Caveats & Limitations", level=2)
            for c in report.caveats:
                doc.add_paragraph(c, style="List Bullet")
            doc.add_paragraph()

        # ── Footer ──
        p = doc.add_paragraph("Generated by Aristostat")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        p.runs[0].font.size = Pt(9)

        # ── Save ──
        project_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir  = os.path.join(project_dir, "..", "Output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "aristostat_report.docx")
        doc.save(output_path)

        print(f"[generate_docx_report] Saved to: {output_path}")

        report.docx_path      = output_path
        report.docx_generated = True
        _report_store["report_output"] = report

        return f"SUCCESS: Report saved to {output_path}"

    except Exception as e:
        import traceback
        print(f"[generate_docx_report] EXCEPTION: {traceback.format_exc()}")
        return f"ERROR: Could not generate docx — {str(e)}"



# ─────────────────────────────────────────────
# EXPORTED TOOL LIST
# ─────────────────────────────────────────────

FINAL_REPORT_TOOLS = [
    get_pipeline_summary,
    build_and_render_report,
    generate_docx_report,
]
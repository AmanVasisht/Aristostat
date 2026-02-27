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
    Returns the file path of the generated .docx on success.
    """
    report: FinalReportOutput | None = _report_store.get("report_output")
    if report is None:
        return "ERROR: No report assembled. Call build_and_render_report first."

    # ── Build Node.js script content ──
    script = _build_docx_script(report)

    try:
        # Write script to temp file and execute
        with tempfile.NamedTemporaryFile(
            suffix=".js", mode="w", delete=False, dir="/home/claude"
        ) as f:
            f.write(script)
            script_path = f.name

        output_path = "/home/claude/aristostat_report.docx"

        result = subprocess.run(
            ["node", script_path],
            capture_output=True, text=True, timeout=30
        )

        os.unlink(script_path)

        if result.returncode != 0:
            return f"ERROR: docx generation failed — {result.stderr}"

        # Copy to outputs
        final_path = "/mnt/user-data/outputs/aristostat_report.docx"
        import shutil
        shutil.copy(output_path, final_path)

        report.docx_path = final_path
        report.docx_generated = True
        _report_store["report_output"] = report

        return f"SUCCESS: Report saved to {final_path}"

    except Exception as e:
        return f"ERROR: Could not generate docx — {str(e)}"


# ─────────────────────────────────────────────
# DOCX SCRIPT BUILDER
# ─────────────────────────────────────────────

def _escape(s: str) -> str:
    """Escapes a string for safe embedding in a JS template literal."""
    return (
        s.replace("\\", "\\\\")
         .replace("`", "\\`")
         .replace("${", "\\${")
    )


def _build_docx_script(report: FinalReportOutput) -> str:
    """
    Builds the Node.js docx-js script that generates the Word document.
    Follows all rules from the docx SKILL.md:
    - US Letter page size (12240 x 15840 DXA)
    - 1-inch margins
    - Arial font
    - Proper bullet numbering (no unicode bullets)
    - Dual table widths (table + cell)
    - ShadingType.CLEAR
    """

    # ── Build caveats bullet items ──
    caveat_items = ""
    for c in report.caveats:
        caveat_items += f"""
      new Paragraph({{
        numbering: {{ reference: "bullets", level: 0 }},
        children: [new TextRun(`{_escape(c)}`)]
      }}),"""

    caveat_section = ""
    if report.caveats:
        caveat_section = f"""
    new Paragraph({{
      heading: HeadingLevel.HEADING_2,
      children: [new TextRun("Caveats & Limitations")]
    }}),
    {caveat_items}
    new Paragraph({{ children: [new TextRun("")] }}),"""

    # ── Build rectifications bullet items ──
    rect_items = ""
    for r in report.rectifications_applied:
        rect_items += f"""
      new Paragraph({{
        numbering: {{ reference: "bullets", level: 0 }},
        children: [new TextRun(`{_escape(r)}`)]
      }}),"""

    rect_section = ""
    if report.rectifications_applied:
        rect_section = f"""
    new Paragraph({{
      heading: HeadingLevel.HEADING_2,
      children: [new TextRun("Rectifications Applied")]
    }}),
    {rect_items}
    new Paragraph({{ children: [new TextRun("")] }}),"""

    post_test_section = ""
    if report.post_test_summary:
        post_test_section = f"""
    new Paragraph({{
      heading: HeadingLevel.HEADING_2,
      children: [new TextRun("Post-Test Model Checks")]
    }}),
    new Paragraph({{ children: [new TextRun(`{_escape(report.post_test_summary)}`)] }}),
    new Paragraph({{ children: [new TextRun("")] }}),"""

    effect_size_section = ""
    if report.effect_size:
        effect_size_section = f"""
    new Paragraph({{
      children: [
        new TextRun({{ text: "Effect size: ", bold: true }}),
        new TextRun(`{_escape(report.effect_size)}`)
      ]
    }}),"""

    script = f"""
const fs = require('fs');
const {{
  Document, Packer, Paragraph, TextRun, HeadingLevel,
  AlignmentType, LevelFormat, BorderStyle
}} = require('docx');

const doc = new Document({{
  numbering: {{
    config: [
      {{
        reference: "bullets",
        levels: [{{
          level: 0,
          format: LevelFormat.BULLET,
          text: "\\u2022",
          alignment: AlignmentType.LEFT,
          style: {{ paragraph: {{ indent: {{ left: 720, hanging: 360 }} }} }}
        }}]
      }}
    ]
  }},
  styles: {{
    default: {{
      document: {{ run: {{ font: "Arial", size: 24 }} }}
    }},
    paragraphStyles: [
      {{
        id: "Heading1", name: "Heading 1", basedOn: "Normal",
        next: "Normal", quickFormat: true,
        run: {{ size: 36, bold: true, font: "Arial", color: "1F3864" }},
        paragraph: {{ spacing: {{ before: 320, after: 160 }}, outlineLevel: 0 }}
      }},
      {{
        id: "Heading2", name: "Heading 2", basedOn: "Normal",
        next: "Normal", quickFormat: true,
        run: {{ size: 28, bold: true, font: "Arial", color: "2E75B6" }},
        paragraph: {{ spacing: {{ before: 240, after: 120 }}, outlineLevel: 1 }}
      }}
    ]
  }},
  sections: [{{
    properties: {{
      page: {{
        size: {{ width: 12240, height: 15840 }},
        margin: {{ top: 1440, right: 1440, bottom: 1440, left: 1440 }}
      }}
    }},
    children: [
      new Paragraph({{
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun(`{_escape(report.title)}`)]
      }}),

      new Paragraph({{
        children: [
          new TextRun({{ text: "Query: ", bold: true }}),
          new TextRun(`{_escape(report.original_query)}`)
        ]
      }}),
      new Paragraph({{ children: [new TextRun("")] }}),

      new Paragraph({{
        border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 6, color: "2E75B6", space: 1 }} }},
        children: [new TextRun("")]
      }}),
      new Paragraph({{ children: [new TextRun("")] }}),

      new Paragraph({{
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("Dataset & Preparation")]
      }}),
      new Paragraph({{ children: [new TextRun(`{_escape(report.dataset_summary)}`)] }}),
      new Paragraph({{ children: [new TextRun("")] }}),

      new Paragraph({{
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("Test Selection")]
      }}),
      new Paragraph({{ children: [new TextRun(`{_escape(report.test_selected)}`)] }}),
      new Paragraph({{ children: [new TextRun("")] }}),

      new Paragraph({{
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("Assumption Checks")]
      }}),
      new Paragraph({{ children: [new TextRun(`{_escape(report.assumptions_summary)}`)] }}),
      new Paragraph({{ children: [new TextRun("")] }}),

      {rect_section}

      {post_test_section}

      new Paragraph({{
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("Results")]
      }}),
      new Paragraph({{
        children: [
          new TextRun({{ text: `{_escape(report.key_statistic)}`, bold: true }})
        ]
      }}),
      new Paragraph({{ children: [new TextRun(`{_escape(report.verdict)}`)] }}),
      {effect_size_section}
      new Paragraph({{ children: [new TextRun("")] }}),

      new Paragraph({{
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("Interpretation")]
      }}),
      new Paragraph({{ children: [new TextRun(`{_escape(report.interpretation)}`)] }}),
      new Paragraph({{ children: [new TextRun("")] }}),

      {caveat_section}

      new Paragraph({{
        border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 6, color: "CCCCCC", space: 1 }} }},
        children: [new TextRun("")]
      }}),
      new Paragraph({{
        alignment: AlignmentType.CENTER,
        children: [new TextRun({{ text: "Generated by Aristostat", color: "888888", size: 18 }})]
      }}),
    ]
  }}]
}});

Packer.toBuffer(doc).then(buffer => {{
  fs.writeFileSync('/home/claude/aristostat_report.docx', buffer);
  console.log('SUCCESS');
}});
"""
    return script


# ─────────────────────────────────────────────
# EXPORTED TOOL LIST
# ─────────────────────────────────────────────

FINAL_REPORT_TOOLS = [
    get_pipeline_summary,
    build_and_render_report,
    generate_docx_report,
]
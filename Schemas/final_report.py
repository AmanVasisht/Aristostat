"""
FILE: schemas/final_report_schema.py
--------------------------------------
Pydantic output schema for the Final Report agent.
FinalReportOutput carries the complete report content and
the path to the generated .docx file.
"""

from pydantic import BaseModel, Field


class FinalReportOutput(BaseModel):
    # ── Report content sections (used for both chat display and docx) ──
    title:               str = ""
    original_query:      str = ""

    # ── Pipeline summary ──
    dataset_summary:     str = ""   # rows, cols, cleaning done
    test_selected:       str = ""   # which test, why it was selected
    assumptions_summary: str = ""   # which passed, which failed, what was done
    rectifications_applied: list[str] = Field(default_factory=list)
    post_test_summary:   str = ""   # model critic results (regression only)

    # ── Core results ──
    key_statistic:       str = ""   # e.g. "t = -3.14, p = 0.0018"
    verdict:             str = ""   # plain English significance verdict
    effect_size:         str = ""   # effect size and label if available
    interpretation:      str = ""   # full plain English interpretation

    # ── Caveats ──
    caveats:             list[str] = Field(default_factory=list)
    # Populated from: accepted violations, warnings, post-test failures proceeded past

    # ── Markdown version (for chat display) ──
    markdown_report:     str = ""

    # ── File path of generated .docx ──
    docx_path:           str = ""
    docx_generated:      bool = False
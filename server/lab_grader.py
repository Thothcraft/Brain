"""Lab grading abstraction.

Notebook execution and LLM grading are deferred behind this interface;
no untrusted notebook is executed by Brain or the Hub in the initial
release. A future backend (isolated Docker workers, Modal, JupyterHub)
implements ``LabGrader`` and is swapped in via ``get_grader()``.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_NOTEBOOK_BYTES = 20 * 1024 * 1024  # 20 MB


# ── result model ──────────────────────────────────────────────────────────────

@dataclass
class GradeResult:
    score: Optional[float] = None
    max_score: Optional[float] = None
    passed: Optional[bool] = None
    feedback: List[str] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    execution_status: str = "not_run"  # not_run | success | error


# ── notebook validation ───────────────────────────────────────────────────────

class NotebookValidationError(ValueError):
    """Raised when an uploaded .ipynb fails structural validation."""


def validate_notebook(raw: bytes) -> Dict[str, Any]:
    """Parse and structurally validate a .ipynb upload.

    Returns extracted metadata: nbformat, cell counts, kernelspec,
    whether outputs are present, and the lab metadata block if the
    template embedded one (``metadata.thothcraft``).
    """
    if not raw:
        raise NotebookValidationError("Empty upload")
    if len(raw) > MAX_NOTEBOOK_BYTES:
        raise NotebookValidationError("Notebook exceeds 20 MB limit")

    try:
        nb = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise NotebookValidationError(f"Not a valid .ipynb (JSON parse failed): {e}")

    if not isinstance(nb, dict) or "cells" not in nb or "nbformat" not in nb:
        raise NotebookValidationError("Missing required nbformat keys (cells, nbformat)")
    if not isinstance(nb["cells"], list):
        raise NotebookValidationError("'cells' must be a list")
    if int(nb.get("nbformat", 0)) < 4:
        raise NotebookValidationError("nbformat < 4 is not supported")

    cells = nb["cells"]
    code_cells = [c for c in cells if isinstance(c, dict) and c.get("cell_type") == "code"]
    md_cells = [c for c in cells if isinstance(c, dict) and c.get("cell_type") == "markdown"]
    has_outputs = any(c.get("outputs") for c in code_cells)
    executed = any(c.get("execution_count") for c in code_cells)

    meta = nb.get("metadata") or {}
    return {
        "nbformat": nb.get("nbformat"),
        "nbformat_minor": nb.get("nbformat_minor"),
        "kernelspec": (meta.get("kernelspec") or {}).get("name"),
        "cell_count": len(cells),
        "code_cells": len(code_cells),
        "markdown_cells": len(md_cells),
        "has_outputs": has_outputs,
        "was_executed": executed,
        "lab_meta": meta.get("thothcraft"),
    }


def check_required_artifacts(
    nb_meta: Dict[str, Any], required: List[str]
) -> List[str]:
    """Return the list of required artifacts missing from the notebook.

    Recognized artifacts: ``notebook``, ``outputs``, ``metrics``,
    ``figures``, ``conclusions``. ``metrics``/``figures``/``conclusions``
    are detected heuristically from cell content until the execution
    backend can extract real artifacts.
    """
    missing: List[str] = []
    for artifact in required:
        if artifact == "notebook":
            continue  # the upload itself satisfies this
        if artifact == "outputs" and not nb_meta.get("has_outputs"):
            missing.append(artifact)
        elif artifact == "metrics" and not nb_meta.get("has_outputs"):
            missing.append(artifact)
        elif artifact == "figures" and not nb_meta.get("has_outputs"):
            missing.append(artifact)
        elif artifact == "conclusions" and nb_meta.get("markdown_cells", 0) == 0:
            missing.append(artifact)
    return missing


# ── grader interface ──────────────────────────────────────────────────────────

class LabGrader(ABC):
    """Grades a lab submission. Implementations must be sandboxed if they
    execute notebook code; Brain itself never executes submissions."""

    @abstractmethod
    def grade(
        self,
        notebook_path: str,
        lab_spec: Dict[str, Any],
        rubric: Optional[Dict[str, Any]],
        nb_meta: Dict[str, Any],
    ) -> GradeResult:
        ...


class StructuralGrader(LabGrader):
    """Initial adapter: structural checks only, no execution, no LLM.

    Produces a provisional result and leaves ``status`` for a human or a
    future execution backend to finalize. Missing required artifacts are
    reported as feedback; score stays None so the submission remains
    'pending' rather than auto-passed.
    """

    def grade(
        self,
        notebook_path: str,
        lab_spec: Dict[str, Any],
        rubric: Optional[Dict[str, Any]],
        nb_meta: Dict[str, Any],
    ) -> GradeResult:
        required = lab_spec.get("required_artifacts") or []
        missing = check_required_artifacts(nb_meta, required)

        feedback: List[str] = []
        if missing:
            feedback.append(
                "Missing required artifacts: " + ", ".join(missing)
            )
        if not nb_meta.get("was_executed"):
            feedback.append(
                "Notebook does not appear to have been executed "
                "(no execution counts). Run all cells before submitting."
            )
        if not feedback:
            feedback.append(
                "Structural checks passed. Awaiting full grading "
                "(execution-based grading is not yet enabled)."
            )

        return GradeResult(
            score=None,
            max_score=lab_spec.get("max_score"),
            passed=None,
            feedback=feedback,
            artifacts=[{"type": "notebook", "path": notebook_path}],
            execution_status="not_run",
        )


def get_grader() -> LabGrader:
    """Return the active grader backend.

    Swap point for a future sandboxed executor (Docker worker VM, Modal,
    JupyterHub). Selected via LAB_GRADER env var; only 'structural' and
    'manual' exist today.
    """
    import os

    backend = os.getenv("LAB_GRADER", "structural")
    if backend == "structural":
        return StructuralGrader()
    logger.warning("Unknown LAB_GRADER '%s'; falling back to structural", backend)
    return StructuralGrader()

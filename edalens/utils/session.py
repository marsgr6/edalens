"""Save an EDALens session to one JSON file and open it again where it was left.

The file holds the loaded data (optional), the operation history (Load data,
Missing Data and Statistics steps, replayed on restore), the Visualization page's
plot and mappings, the custom category orders and the report.
"""

from __future__ import annotations

import io
import json
from collections.abc import MutableMapping
from typing import Any, Optional

import pandas as pd

from edalens.utils.data_processing import OperationError, apply_operations, set_working_df
from edalens.utils.report import Report

FORMAT = "edalens-session"
FORMAT_VERSION = 1
# Widget keys the Visualization page fills from viz_state when they are missing.
PLOT_WIDGET_KEYS = ("_plot_type_selector", "x_zone", "y_zone", "color_zone", "facet_col_zone",
                    "facet_row_zone", "size_zone")


class SessionError(ValueError):
    """The file is not an EDALens session, or it cannot be applied to the data."""


def export_session(state: MutableMapping[str, Any], include_data: bool = True) -> str:
    """The session as JSON text."""
    if "raw_df" not in state:
        raise SessionError("Load data first: there is nothing to save yet.")
    from edalens import __version__

    report = state.get("report")
    payload = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "edalens": __version__,
        "source": state.get("last_uploaded_file"),
        "columns": [str(c) for c in state["raw_df"].columns],
        "data": state["raw_df"].to_json(orient="table", date_format="iso") if include_data else None,
        "operations": [dict(op) for op in state.get("operations", [])],
        "enable_operation_tracking": bool(state.get("enable_operation_tracking", False)),
        "viz_state": state.get("viz_state"),
        "viz_memory": state.get("viz_memory"),
        "category_orders": state.get("category_orders") or {},
        "report": report.to_json() if report is not None else None,
    }
    return json.dumps(payload, default=str)


def read_session(text: str | bytes) -> dict:
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise SessionError("This file is not an EDALens session (not JSON).") from exc
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise SessionError("This file is not an EDALens session.")
    if payload.get("format_version", 0) > FORMAT_VERSION:
        raise SessionError("This session was saved by a newer EDALens; upgrade edalens to open it.")
    return payload


def restore_session(state: MutableMapping[str, Any], text: str | bytes,
                    data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Rebuild the session from ``text``; returns the working data.

    Without data in the file, ``data`` (or the data already loaded) is used; its
    columns must match the saved ones. Raises SessionError and leaves ``state``
    unchanged if the operations cannot be replayed.
    """
    payload = read_session(text)
    if payload.get("data"):
        raw = pd.read_json(io.StringIO(payload["data"]), orient="table")
    else:
        raw = data if data is not None else state.get("raw_df")
        if raw is None:
            raise SessionError("This session was saved without its data: load the same data first, then open it.")
        missing = [c for c in payload.get("columns", []) if c not in map(str, raw.columns)]
        if missing:
            raise SessionError(f"The loaded data lacks columns the session uses: {', '.join(missing)}.")
    operations = [dict(op) for op in payload.get("operations", [])]
    try:
        working = apply_operations(raw, operations)
    except OperationError as exc:
        raise SessionError(f"The saved steps could not be replayed: {exc}") from exc

    for key in PLOT_WIDGET_KEYS + ("imputed_df",):
        state.pop(key, None)
    state["raw_df"] = raw.copy()
    set_working_df(state, working, operations)
    state["data_loaded"] = True
    state["last_uploaded_file"] = payload.get("source") or "Restored session"
    state["enable_operation_tracking"] = payload.get("enable_operation_tracking", False)
    if payload.get("viz_state"):
        state["viz_state"] = payload["viz_state"]
        state["_plot_type_selector"] = payload["viz_state"].get("plot_type", "bars")
    if payload.get("viz_memory"):
        state["viz_memory"] = payload["viz_memory"]
    # Custom orders belong to this data version (see order_select on the Visualization page).
    state["category_orders"] = payload.get("category_orders") or {}
    state["category_orders_version"] = state["data_version"]
    if payload.get("report"):
        state["report"] = Report.from_json(payload["report"])
    return working

"""HTML-based table renderer with per-cell tooltips.

Streamlit's native ``st.dataframe`` only supports column-header tooltips via
``column_config(...help=...)`` — cell contents get truncated with no way to
reveal the full text or contextual drill-down on hover. This helper renders
a table as raw HTML so every cell can carry a ``title`` attribute that
browsers expose as a native hover tooltip.

Usage:
    from app.utils.tooltip_html import render_tooltip_table

    render_tooltip_table(
        rows=[{'Pri': '#1', 'Current': 'Luis Campusano (R)', …}, …],
        columns=['Pri', 'Current', 'Meta', 'Confidence'],
        header_help={
            'Pri': 'Priority rank within role',
            'Meta': 'Calibrated meta score',
        },
        cell_tooltip={
            'Current': lambda r: f"{r['_full_name']} · {r['_card_type']} · tier {r['_tier']}",
            'Meta': lambda r: r.get('_meta_breakdown'),
        },
    )

Design notes:
    * Uses Streamlit's dark-theme-friendly palette so tables blend in.
    * ``cell_tooltip`` callbacks receive the full row dict, so they can
      pull extra fields prefixed with '_' that wouldn't normally be shown.
    * Long tooltip strings wrap automatically in the browser.
    * No JavaScript required — this is pure HTML title attributes.
"""
from __future__ import annotations

import html
from typing import Callable, Optional

import streamlit as st


_TOOLTIP_CSS = """
<style>
.tt-table { width: 100%; border-collapse: collapse; font-size: 14px;
            color: rgba(250, 250, 250, 0.92); }
.tt-table thead th { text-align: left; font-weight: 600;
                     padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,0.14);
                     color: rgba(250, 250, 250, 0.75); position: sticky; top: 0;
                     background-color: #0e1117; z-index: 1; }
.tt-table tbody td { padding: 7px 10px; border-bottom: 1px solid rgba(255,255,255,0.08);
                     vertical-align: middle; }
.tt-table tbody tr:hover { background-color: rgba(255,255,255,0.03); }
.tt-cell-help { border-bottom: 1px dotted rgba(120,140,170,0.5); cursor: help; }
.tt-small { color: rgba(250,250,250,0.55); font-size: 12px; }
</style>
"""
_CSS_INJECTED_KEY = '_tt_css_injected'


def _ensure_css() -> None:
    """Inject the tooltip CSS once per Streamlit session."""
    if st.session_state.get(_CSS_INJECTED_KEY):
        return
    st.markdown(_TOOLTIP_CSS, unsafe_allow_html=True)
    st.session_state[_CSS_INJECTED_KEY] = True


def _tip(text: str) -> str:
    """Escape for HTML title attribute. Newlines become &#10; so they render."""
    if text is None:
        return ''
    s = str(text)
    # Collapse long whitespace runs but keep \n as literal newlines in the tooltip
    return html.escape(s, quote=True).replace('\n', '&#10;')


def render_tooltip_table(
    rows: list[dict],
    columns: list[str],
    *,
    header_help: Optional[dict[str, str]] = None,
    cell_tooltip: Optional[dict[str, Callable[[dict], str]]] = None,
    max_height_px: int = 600,
) -> None:
    """Render a scrollable HTML table with per-cell + per-header tooltips.

    Args:
        rows: list of dicts. Keys not in ``columns`` are ignored for display
            but may still be accessed by ``cell_tooltip`` callbacks.
        columns: column ordering + names to display.
        header_help: ``{col_name: tooltip_text}`` for column headers.
        cell_tooltip: ``{col_name: row_dict → tooltip_text}``. Return value
            is used as the cell's ``title`` attribute. Return None/'' to
            skip the tooltip for that cell.
        max_height_px: scrollbox height.
    """
    _ensure_css()
    header_help = header_help or {}
    cell_tooltip = cell_tooltip or {}

    parts: list[str] = [
        f'<div style="max-height:{max_height_px}px; overflow-y:auto;">'
        '<table class="tt-table"><thead><tr>'
    ]
    for col in columns:
        tip = header_help.get(col)
        if tip:
            parts.append(f'<th title="{_tip(tip)}"><span class="tt-cell-help">{html.escape(str(col))}</span></th>')
        else:
            parts.append(f'<th>{html.escape(str(col))}</th>')
    parts.append('</tr></thead><tbody>')

    for r in rows:
        parts.append('<tr>')
        for col in columns:
            cell = r.get(col, '')
            cell_str = '' if cell is None else str(cell)
            # Tooltip builder — called with full row dict
            tip = ''
            builder = cell_tooltip.get(col)
            if builder is not None:
                try:
                    t = builder(r)
                    tip = '' if t is None else str(t)
                except Exception:
                    tip = ''
            if tip:
                parts.append(
                    f'<td title="{_tip(tip)}">'
                    f'<span class="tt-cell-help">{html.escape(cell_str)}</span>'
                    f'</td>'
                )
            else:
                parts.append(f'<td>{html.escape(cell_str)}</td>')
        parts.append('</tr>')

    parts.append('</tbody></table></div>')
    st.markdown(''.join(parts), unsafe_allow_html=True)


def tooltip_span(text: str, tip: str) -> str:
    """Inline helper — wrap a piece of text with a native-tooltip span.

    Usable inside st.markdown(..., unsafe_allow_html=True). The visible
    text gets a dotted underline so users know it's hoverable.
    """
    _ensure_css()
    if not tip:
        return html.escape(str(text))
    return (f'<span class="tt-cell-help" title="{_tip(tip)}">'
            f'{html.escape(str(text))}</span>')

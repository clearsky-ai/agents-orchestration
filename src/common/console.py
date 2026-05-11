"""Small console pretty-printer for the orchestration flow.

ANSI colors auto-disable when stdout is not a TTY (e.g. piped to a file).
No external deps — keep it deliberately minimal.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import textwrap
from typing import Any, Mapping


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


_COLOR = _supports_color()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def bold(s: str) -> str:
    return _c("1", s)


def dim(s: str) -> str:
    return _c("2", s)


def cyan(s: str) -> str:
    return _c("36", s)


def green(s: str) -> str:
    return _c("32", s)


def yellow(s: str) -> str:
    return _c("33", s)


def magenta(s: str) -> str:
    return _c("35", s)


def _term_width(default: int = 100) -> int:
    try:
        return max(60, min(shutil.get_terminal_size((default, 20)).columns, 120))
    except Exception:
        return default


def banner(title: str, color=cyan) -> None:
    width = _term_width()
    bar = "=" * width
    print(color(bar), flush=True)
    print(color(bold(f"  {title}")), flush=True)
    print(color(bar), flush=True)


def section(title: str, color=cyan) -> None:
    width = _term_width()
    pad = max(0, width - len(title) - 4)
    line = f"-- {title} " + ("-" * pad)
    print(color(line[:width]), flush=True)


def body(text: str, indent: int = 2) -> None:
    width = _term_width() - indent
    prefix = " " * indent
    for line in text.splitlines() or [""]:
        if not line.strip():
            print("", flush=True)
            continue
        wrapped = textwrap.wrap(
            line,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        ) or [line]
        for w in wrapped:
            print(prefix + w, flush=True)


def kv(label: str, value: str, label_color=yellow) -> None:
    print(f"  {label_color(label + ':')} {value}", flush=True)


def progress(label: str, current: int, total: int) -> None:
    filled = "#" * current + "." * max(0, total - current)
    print(
        f"  {dim('[' + filled + ']')} {label} {current}/{total}",
        flush=True,
    )


def pretty_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except TypeError:
        return str(obj)


def final_answer_box(title: str, text: str) -> None:
    """Render the orchestrator's final answer in a bordered box."""
    width = _term_width()
    top = "+" + "-" * (width - 2) + "+"
    side = "|"
    print(green(top), flush=True)
    title_line = f" {bold(title)} "
    print(green(side) + title_line.ljust(width - 2) + green(side), flush=True)
    print(green("|" + "-" * (width - 2) + "|"), flush=True)
    inner = width - 4
    for raw in text.splitlines() or [""]:
        if not raw.strip():
            print(green(side) + " " * (width - 2) + green(side), flush=True)
            continue
        wrapped = textwrap.wrap(
            raw, width=inner, break_long_words=False, break_on_hyphens=False
        ) or [raw]
        for w in wrapped:
            print(green(side) + " " + w.ljust(inner) + " " + green(side), flush=True)
    print(green(top), flush=True)


def render_plans(plans: Mapping[str, str]) -> None:
    for agent, plan in plans.items():
        section(f"plan -> {agent}", color=magenta)
        body(plan)

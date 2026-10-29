"""Prevent regressions in the portable math syntax used by repository viewers."""

import re
from pathlib import Path

import pytest


@pytest.mark.parametrize("relative_path", ["README.md", "docs/research.md"])
def test_display_math_preserves_subscripts_and_uses_portable_macros(relative_path):
    text = (Path(__file__).resolve().parents[1] / relative_path).read_text(encoding="utf-8")
    delimiters = [line for line in text.splitlines() if "$$" in line]
    assert delimiters, "Research documentation must retain its mathematical formulation"
    assert all(line.strip() == "$$" for line in delimiters), (
        "Keep $$ on standalone lines so Markdown does not consume math underscores"
    )
    assert len(delimiters) % 2 == 0, "Every display math block needs a closing delimiter"
    for expression in re.findall(r"\$\$(.*?)\$\$", text, re.DOTALL):
        assert r"\operatorname" not in expression, "Use portable roman text or explicit formulas"
        assert r"\_" not in expression, "Math subscripts use _, not Markdown-escaped underscores"

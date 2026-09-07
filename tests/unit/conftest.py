"""Shared test configuration."""

import os

# Typer renders usage errors through Rich, which forces color when GITHUB_ACTIONS
# is set and then splits option names into separately styled runs, so the literal
# "--input-format" never appears in the output. Pin plain text and a wide console
# so CLI output is identical locally and in CI. Both are read by typer.rich_utils
# at import time, hence the assignment here rather than in a fixture.
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"
os.environ.setdefault("TERMINAL_WIDTH", "200")

"""Entry point for ``python -m greenai``.

Useful when the installed ``greenai`` script directory is not on PATH.
"""

from __future__ import annotations

from greenai.cli import app

if __name__ == "__main__":
    app()

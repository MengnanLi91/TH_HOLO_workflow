"""Summarize direct and coupled pressure-drop comparison artifacts.

This is the current workflow entry point. The legacy
``summarize_claim_evidence`` module remains available for archived runbooks.
"""

from cases.alpha_d.summarize_claim_evidence import main, summarize_study

__all__ = ["main", "summarize_study"]


if __name__ == "__main__":
    raise SystemExit(main())

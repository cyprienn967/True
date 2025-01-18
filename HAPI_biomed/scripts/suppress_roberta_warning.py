# File: HAPI/scripts/suppress_roberta_warning.py

import sys
import re
import io
from contextlib import contextmanager

@contextmanager
def suppress_roberta_warning():
    """
    A context manager that captures sys.stderr, filters out lines containing:
      - "Some weights of RobertaModel..."
      - "You should probably TRAIN this model..."
    and prints any remaining lines.
    """
    original_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        captured = sys.stderr.getvalue()
        sys.stderr = original_stderr

        # Filter out lines containing the unwanted patterns
        lines = captured.splitlines()
        filtered = []
        for ln in lines:
            if ("Some weights of RobertaModel" in ln) or ("You should probably TRAIN this model" in ln):
                # skip
                continue
            filtered.append(ln)
        
        # Print back any leftover lines
        if filtered:
            print("\n".join(filtered), file=sys.stderr)

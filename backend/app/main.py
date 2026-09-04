"""Backward-compatible import path for the refactored TraceLens app."""

import sys

from tracelens import api as _api

# Existing commands and tests can keep importing app.main while the
# implementation lives under the tracelens package.
sys.modules[__name__] = _api

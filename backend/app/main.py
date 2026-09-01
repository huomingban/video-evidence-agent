"""Backward-compatible import path for the refactored SeeIt-style app."""

import sys

from seeit import api as _api

# Existing commands and tests can keep importing app.main while the
# implementation lives under the learning-oriented seeit package.
sys.modules[__name__] = _api

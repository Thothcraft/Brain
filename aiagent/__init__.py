"""aiagent package initialization.

This package contains modules for AI assistant functionality including:
- handler: API endpoints and query handlers.
- memory: memory management utilities.
- context: reference file utilities.
- functions: function registry and schemas for tool-calling.

"""

# Make key classes easily accessible at package level
from importlib import import_module

# Re-export FunctionsRegistry at top level for convenience
_functions_mod = import_module("aiagent.functions")
FunctionsRegistry = getattr(_functions_mod, "FunctionsRegistry")

__all__ = [
    "FunctionsRegistry",
]

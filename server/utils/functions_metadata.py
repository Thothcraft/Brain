"""Compatibility shim for older imports.

Earlier code referenced `server.utils.functions_metadata.function_schema` but this
module was later moved to `aiagent.functions.metadata`.  To avoid touching all
call-sites, we provide this thin proxy that re-exports the same symbol.
"""

from importlib import import_module

# Import the canonical implementation
_function_mod = import_module("aiagent.functions.metadata")
function_schema = getattr(_function_mod, "function_schema")

__all__ = ["function_schema"]

# ---------------------------------------------------------------------------
# Legacy import support: map this module to 'utils.functions_metadata'
# ---------------------------------------------------------------------------
import sys
sys.modules.setdefault("utils", import_module("types").ModuleType("utils"))  # create dummy pkg if absent
sys.modules["utils.functions_metadata"] = sys.modules[__name__]

"""Brain v1 public API - the versioned contract all clients consume.

Whispy, Thoth, thothHUB, and the Flutter mobile app all speak to
``/v1/...``. Breaking changes require a new version namespace rather
than silently changing behavior (Architecture v3.0 section 2, section 9.2).
"""

from .router import router

__all__ = ["router"]

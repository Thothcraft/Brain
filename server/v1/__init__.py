"""Brain v1 public API - the versioned contract all clients consume.

Whispy, Thoth, thothHUB, and the Flutter mobile app all speak to
``/v1/...``. Breaking changes require a new version namespace rather
than silently changing behavior (Architecture v3.0 section 2, section 9.2).
"""

from .router import router
from .context import router as context_router
from .automation import router as automation_router
from .faces import router as faces_router
from .subscriptions import router as subscriptions_router
from server.endpoints.node_ws import router as node_ws_router

router.include_router(context_router)
router.include_router(automation_router)
router.include_router(faces_router)
router.include_router(subscriptions_router)
router.include_router(node_ws_router)

__all__ = ["router"]

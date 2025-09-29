"""Main API Routes Module - Modular Endpoint Integration.

This module serves as the central router that imports and integrates
all modularized endpoint modules.

All endpoint logic has been moved to dedicated modules:
- System endpoints: server/endpoints/system_endpoints.py
- Authentication: server/endpoints/auth_endpoints.py  
- AI/Query processing: server/endpoints/ai_endpoints.py
- Device management: server/endpoints/device_endpoints.py
- Data operations: server/endpoints/data_endpoints.py
- File management: server/endpoints/file_endpoints.py
- Webhooks: server/endpoints/webhook_endpoints.py
- Shared models: server/endpoints/models.py
"""

from fastapi import APIRouter

# Import all endpoint routers
from server.endpoints.system_endpoints import router as system_router
from server.endpoints.auth_endpoints import router as auth_router
from server.endpoints.ai_endpoints import router as ai_router
from server.endpoints.device_endpoints import router as device_router
from server.endpoints.data_endpoints import router as data_router
from server.endpoints.file_endpoints import router as file_router
from server.endpoints.webhook_endpoints import router as webhook_router
from server.endpoints.sensor_endpoints import router as sensor_router
from server.endpoints.network_endpoints import router as network_router
from server.endpoints.training_endpoints import router as training_router
from server.endpoints.curriculum_endpoints import router as curriculum_router

# Create main router
router = APIRouter()

# Include all endpoint routers
router.include_router(system_router)      # /, /health
router.include_router(auth_router)        # /token, /register, /profile
router.include_router(ai_router)          # /query
router.include_router(device_router)      # /device/*
router.include_router(data_router)        # /data/*
router.include_router(file_router)        # /file/*
router.include_router(webhook_router)     # /phone/* (Twilio webhooks)
router.include_router(sensor_router)      # /sensors/* (Sense HAT sensors)
router.include_router(network_router)     # /network/* (WiFi configuration)
router.include_router(training_router)    # /training/*, /federated/* (ML training)
router.include_router(curriculum_router)  # /curriculum/* (Education content)

# ============================================================================
# MODULAR ENDPOINTS LOADED
# ============================================================================
# 
# All endpoints are now organized in focused modules:
#
# 📁 system_endpoints.py    - System health and info
# 📁 auth_endpoints.py      - Authentication and user management
# 📁 ai_endpoints.py        - AI query processing
# 📁 device_endpoints.py    - Device registration and management
# 📁 data_endpoints.py      - Data upload and analytics
# 📁 file_endpoints.py      - File upload and management
# 📁 webhook_endpoints.py   - Phone/Twilio webhooks (/phone/*)
# 📁 sensor_endpoints.py    - Sense HAT sensor management (/sensors/*)
# 📁 network_endpoints.py   - WiFi and network configuration (/network/*)
# 📁 training_endpoints.py  - ML training and federated learning (/training/*, /federated/*)
# 📁 curriculum_endpoints.py - Educational content and progress (/curriculum/*)
# 📁 models.py             - Shared request/response models
#
# Total: 50+ endpoints across 11 focused modules
# ============================================================================

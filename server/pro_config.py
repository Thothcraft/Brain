"""Supabase Pro Plan optimized configuration.

This module contains settings optimized for Supabase Pro Plan performance.
"""

import os
from typing import Dict, Any

class SupabaseProConfig:
    """Configuration optimized for Supabase Pro Plan."""
    
    # Database connection settings for Pro Plan
    # Railway currently runs one API process. A 50 + 100 connection pool is
    # counterproductive with Supabase's transaction pooler and amplifies a
    # reconnect event. Keep the local pool deliberately small and configurable.
    DATABASE_POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "5"))
    DATABASE_MAX_OVERFLOW = int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))
    DATABASE_POOL_TIMEOUT = int(os.getenv("DATABASE_POOL_TIMEOUT", "10"))
    DATABASE_POOL_RECYCLE = int(os.getenv("DATABASE_POOL_RECYCLE", "300"))
    DATABASE_CONNECT_TIMEOUT = int(os.getenv("DATABASE_CONNECT_TIMEOUT", "10"))
    
    # Query optimization settings
    STATEMENT_TIMEOUT = int(os.getenv("DATABASE_STATEMENT_TIMEOUT_MS", "15000"))
    IDLE_TRANSACTION_TIMEOUT = int(os.getenv("DATABASE_IDLE_TRANSACTION_TIMEOUT_MS", "30000"))
    
    # Connection pooling settings
    CONNECTION_PRE_PING = True
    APPLICATION_NAME = "thoth_pro"
    
    # Performance monitoring
    ENABLE_QUERY_LOGGING = True
    SLOW_QUERY_THRESHOLD = 1000  # milliseconds
    
    # Cache settings for Pro Plan
    ENABLE_RESULT_CACHING = True
    CACHE_TTL = 300  # 5 minutes
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get optimized database URL for Pro Plan."""
        base_url = os.getenv("DATABASE_URL")
        if not base_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        # Ensure Pro Plan optimizations are in URL
        if "sslmode=require" not in base_url:
            base_url += "?sslmode=require"
        
        return base_url
    
    @classmethod
    def get_connect_args(cls) -> Dict[str, Any]:
        """Get optimized connection arguments for Pro Plan."""
        return {
            "connect_timeout": cls.DATABASE_CONNECT_TIMEOUT,
            "application_name": cls.APPLICATION_NAME,
            "sslmode": "require",
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
            "options": f"-c statement_timeout={cls.STATEMENT_TIMEOUT} -c idle_in_transaction_session_timeout={cls.IDLE_TRANSACTION_TIMEOUT}"
        }
    
    @classmethod
    def get_engine_kwargs(cls) -> Dict[str, Any]:
        """Get optimized engine kwargs for Pro Plan."""
        return {
            "pool_size": cls.DATABASE_POOL_SIZE,
            "max_overflow": cls.DATABASE_MAX_OVERFLOW,
            "pool_timeout": cls.DATABASE_POOL_TIMEOUT,
            "pool_pre_ping": cls.CONNECTION_PRE_PING,
            "pool_recycle": cls.DATABASE_POOL_RECYCLE,
            "pool_use_lifo": True,
            "connect_args": cls.get_connect_args()
        }

# Global Pro Plan config instance
pro_config = SupabaseProConfig()

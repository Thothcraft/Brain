"""
Database Health Monitor Service

Provides continuous monitoring and automatic recovery for database connections.
This service runs in the background to ensure database connectivity is maintained.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DisconnectionError
from sqlalchemy import text

from server.db import engine, SessionLocal, test_database_connection

logger = logging.getLogger('db_health_monitor')

class DatabaseHealthMonitor:
    """Monitors database health and performs automatic recovery."""
    
    def __init__(self, check_interval: int = 60, max_failures: int = 3):  # Reduced interval for Pro Plan
        self.check_interval = check_interval  # seconds
        self.max_failures = max_failures
        self.failure_count = 0
        self.last_check = None
        self.is_running = False
        self.monitor_task = None
        self.status_history = []
        
        # Try to load Pro Plan configuration
        try:
            from server.pro_config import pro_config
            self.pro_config = pro_config
            logger.info("Database health monitor using Pro Plan configuration")
        except ImportError:
            self.pro_config = None
            logger.info("Database health monitor using standard configuration")
        
    async def start_monitoring(self):
        """Start the background health monitoring task."""
        if self.is_running:
            logger.warning("Database health monitor is already running")
            return
            
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Database health monitor started")
        
    async def stop_monitoring(self):
        """Stop the background health monitoring task."""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Database health monitor stopped")
        
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                await self._check_database_health()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in health monitor: {e}")
                await asyncio.sleep(min(self.check_interval, 30))  # Shorter sleep on error
                
    async def _check_database_health(self):
        """Perform a single health check."""
        try:
            # Test basic connectivity
            status = test_database_connection()
            
            # Test with a more complex query
            with SessionLocal() as db:
                result = db.execute(text("SELECT count(*) as user_count FROM user_account"))
                user_count = result.scalar()
                status['user_count'] = user_count
                
                # Test connection pool stats
                pool = engine.pool
                status.update({
                    'pool_size': pool.size(),
                    'pool_checked_in': pool.checkedin(),
                    'pool_checked_out': pool.checkedout(),
                    'pool_overflow': pool.overflow()
                })
            
            # Reset failure count on success
            if self.failure_count > 0:
                logger.info(f"Database connection restored after {self.failure_count} failures")
                self.failure_count = 0
                
            status['timestamp'] = datetime.utcnow().isoformat()
            status['status'] = 'healthy'
            
            # Keep only last 100 status updates
            self.status_history.append(status)
            if len(self.status_history) > 100:
                self.status_history.pop(0)
                
            self.last_check = datetime.utcnow()
            
        except Exception as e:
            self.failure_count += 1
            error_str = str(e)
            is_ssl_error = "SSL" in error_str or "closed unexpectedly" in error_str
            
            error_status = {
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'unhealthy',
                'error': error_str,
                'error_type': 'ssl_connection' if is_ssl_error else 'general',
                'failure_count': self.failure_count,
                'consecutive_failures': self.failure_count
            }
            
            self.status_history.append(error_status)
            logger.error(f"Database health check failed (attempt {self.failure_count}/{self.max_failures}): {e}")
            
            # For SSL errors, immediately attempt recovery
            if is_ssl_error:
                logger.warning("SSL connection error detected in health check, forcing immediate recovery")
                await self._attempt_recovery()
            elif self.failure_count <= self.max_failures:
                await self._attempt_recovery()
                
    async def _attempt_recovery(self):
        """Attempt to recover database connection."""
        logger.info("Attempting database connection recovery...")
        
        try:
            # Dispose of the engine and all connections
            engine.dispose()
            logger.info("Database engine disposed")
            
            # Wait a moment before attempting reconnection
            await asyncio.sleep(2)
            
            # Test the new connection
            status = test_database_connection()
            if status['status'] == 'connected':
                logger.info("Database connection recovery successful")
                self.failure_count = 0
            else:
                logger.warning(f"Recovery attempt failed: {status.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Database recovery attempt failed: {e}")
            
    def get_status(self) -> Dict:
        """Get current monitor status and recent history."""
        return {
            'is_running': self.is_running,
            'check_interval': self.check_interval,
            'failure_count': self.failure_count,
            'max_failures': self.max_failures,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'recent_status': self.status_history[-10:] if self.status_history else []
        }
        
    async def force_health_check(self) -> Dict:
        """Force an immediate health check."""
        await self._check_database_health()
        return self.status_history[-1] if self.status_history else {}

# Global monitor instance
health_monitor = DatabaseHealthMonitor()

async def start_database_health_monitor():
    """Start the global database health monitor."""
    await health_monitor.start_monitoring()

async def stop_database_health_monitor():
    """Stop the global database health monitor."""
    await health_monitor.stop_monitoring()

def get_database_health_status() -> Dict:
    """Get the current database health status."""
    return health_monitor.get_status()

async def force_database_health_check() -> Dict:
    """Force a database health check."""
    return await health_monitor.force_health_check()

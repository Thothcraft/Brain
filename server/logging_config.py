"""Enhanced logging configuration for debugging and monitoring."""

import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Enhanced logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "[{asctime}] {levelname} [{name}:{lineno}] - {message}",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "style": "{"
        },
        "simple": {
            "format": "{levelname} - {message}",
            "style": "{"
        },
        "json": {
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "detailed",
            "stream": sys.stdout
        },
        "console_detailed": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "stream": sys.stdout
        },
        "file_debug": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": log_dir / "debug.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
        "file_error": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": log_dir / "error.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
        "file_api": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "detailed",
            "filename": log_dir / "api.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
        "file_database": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": log_dir / "database.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
        "file_performance": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "detailed",
            "filename": log_dir / "performance.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        }
    },
    "loggers": {
        "": {  # Root logger
            "level": "INFO",
            "handlers": ["console", "file_debug", "file_error"]
        },
        "server": {  # Application logger
            "level": "DEBUG",
            "handlers": ["console_detailed", "file_debug", "file_error"],
            "propagate": False
        },
        "server.endpoints": {  # API endpoints
            "level": "DEBUG",
            "handlers": ["console_detailed", "file_api", "file_debug"],
            "propagate": False
        },
        "server.db": {  # Database operations
            "level": "DEBUG",
            "handlers": ["file_database", "file_debug"],
            "propagate": False
        },
        "server.services": {  # Background services
            "level": "DEBUG",
            "handlers": ["file_debug", "console"],
            "propagate": False
        },
        "sqlalchemy": {  # SQLAlchemy
            "level": "WARNING",
            "handlers": ["file_database"],
            "propagate": False
        },
        "sqlalchemy.engine": {  # SQLAlchemy engine
            "level": "INFO",
            "handlers": ["file_database"],
            "propagate": False
        },
        "sqlalchemy.pool": {  # Connection pool
            "level": "INFO",
            "handlers": ["file_database"],
            "propagate": False
        },
        "apscheduler": {  # Scheduler
            "level": "DEBUG",
            "handlers": ["file_debug", "console"],
            "propagate": False
        },
        "uvicorn": {  # Uvicorn server
            "level": "INFO",
            "handlers": ["console", "file_debug"],
            "propagate": False
        },
        "fastapi": {  # FastAPI
            "level": "INFO",
            "handlers": ["console", "file_api"],
            "propagate": False
        }
    }
}

def setup_logging():
    """Configure logging with the above configuration."""
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("ThothCraft Server Starting Up")
    logger.info(f"Log level: DEBUG")
    logger.info(f"Log directory: {log_dir.absolute()}")
    logger.info("=" * 80)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)

# Performance monitoring decorator
def log_performance(func_name: str = None):
    """Decorator to log function performance."""
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("performance")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(
                    f"PERFORMANCE: {func_name or func.__name__} executed in {execution_time:.3f}s"
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                logger.error(
                    f"PERFORMANCE: {func_name or func.__name__} failed after {execution_time:.3f}s - {str(e)}"
                )
                
                raise
        
        return wrapper
    return decorator

# Database connection monitoring
class DatabaseConnectionLogger:
    """Monitor database connection health."""
    
    def __init__(self):
        self.logger = logging.getLogger("database")
        self.connection_count = 0
        self.error_count = 0
        
    def log_connection_created(self):
        """Log when a new connection is created."""
        self.connection_count += 1
        self.logger.info(f"DB Connection #{self.connection_count} created")
        
    def log_connection_closed(self):
        """Log when a connection is closed."""
        self.logger.info(f"DB Connection closed (total: {self.connection_count})")
        
    def log_connection_error(self, error: Exception):
        """Log database connection errors."""
        self.error_count += 1
        self.logger.error(f"DB Connection error #{self.error_count}: {str(error)}")
        
    def get_stats(self):
        """Get connection statistics."""
        return {
            "total_connections": self.connection_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(self.connection_count, 1)
        }

# Global instance
db_logger = DatabaseConnectionLogger()

# Request logging middleware helper
def log_request_details(request, response=None, error=None):
    """Log detailed request information."""
    logger = logging.getLogger("api")
    
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "client": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
    }
    
    if response:
        log_data.update({
            "status_code": response.status_code,
            "response_time": getattr(response, 'response_time', None)
        })
        logger.info(f"Request completed: {request.method} {request.url} - {response.status_code}")
    
    elif error:
        logger.error(f"Request failed: {request.method} {request.url} - {str(error)}")
    
    else:
        logger.info(f"Request started: {request.method} {request.url}")

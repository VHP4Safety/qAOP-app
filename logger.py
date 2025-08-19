"""
Structured logging configuration for qAOP Flask application.
Supports JSON format with customizable handlers.
"""
import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any
from config import get_config

config = get_config()

class JSONFormatter(logging.Formatter):
    """Custom formatter for JSON structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)

def setup_logging():
    """Set up structured logging for the application."""
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if config.DEBUG else logging.INFO)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with JSON formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if config.DEBUG else logging.INFO)
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    
    return root_logger

def log_request(request_data: Dict[str, Any], endpoint: str):
    """Log incoming API request."""
    logger = logging.getLogger("qAOP.api")
    logger.info("API request received", extra={
        'extra_fields': {
            'endpoint': endpoint,
            'request_data': request_data,
            'event_type': 'api_request'
        }
    })

def log_simulation_start(model_type: str, dose: float, time: float, simulation_count: int):
    """Log simulation start."""
    logger = logging.getLogger("qAOP.simulation")
    logger.info("Simulation started", extra={
        'extra_fields': {
            'model_type': model_type,
            'dose': dose,
            'time': time,
            'simulation_count': simulation_count,
            'event_type': 'simulation_start'
        }
    })

def log_simulation_complete(model_type: str, duration: float, success: bool):
    """Log simulation completion."""
    logger = logging.getLogger("qAOP.simulation")
    level = logging.INFO if success else logging.ERROR
    message = "Simulation completed successfully" if success else "Simulation failed"
    
    logger.log(level, message, extra={
        'extra_fields': {
            'model_type': model_type,
            'duration': duration,
            'success': success,
            'event_type': 'simulation_complete'
        }
    })

def log_error(error: Exception, context: Dict[str, Any] = None):
    """Log application errors with context."""
    logger = logging.getLogger("qAOP.error")
    logger.error(f"Application error: {str(error)}", extra={
        'extra_fields': {
            'error_type': type(error).__name__,
            'context': context or {},
            'event_type': 'application_error'
        }
    }, exc_info=True)

def log_performance_metric(metric_name: str, value: float, unit: str, context: Dict[str, Any] = None):
    """Log performance metrics."""
    logger = logging.getLogger("qAOP.metrics")
    logger.info(f"Performance metric: {metric_name}", extra={
        'extra_fields': {
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            'context': context or {},
            'event_type': 'performance_metric'
        }
    })
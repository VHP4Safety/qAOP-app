"""
Configuration management for qAOP Flask application.
Supports environment variables with sensible defaults.
"""
import os
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed

class Config:
    """Base configuration class with common settings."""
    
    # Flask settings
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 'yes', 'on')
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('FLASK_PORT', '5000'))
    
    # Application settings
    SIMULATION_COUNT = int(os.getenv('SIMULATION_COUNT', '250'))
    TIME_POINTS = int(os.getenv('TIME_POINTS', '100'))
    
    # Data file paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    STATIC_DIR = BASE_DIR / 'static'
    TEMPLATES_DIR = BASE_DIR / 'templates'
    
    # CSV file paths
    INVITRO_PARAMS_FILE = DATA_DIR / 'draws_cisDDNEC2024-08-05_11-21-18.csv'
    INVIVO_PARAMS_FILE = DATA_DIR / 'draws_cisvivo2024-08-07_10-34-00.csv'
    PK_SUMMARY_FILE = DATA_DIR / 'fit_pk_summary.csv'

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    SIMULATION_COUNT = 10  # Fewer simulations for faster tests
    TIME_POINTS = 20

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on FLASK_ENV environment variable."""
    env = os.getenv('FLASK_ENV', 'default')
    return config.get(env, config['default'])
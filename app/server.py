#!/usr/bin/env python3
"""
Waitress WSGI Server for FTT-ML Backend
Alternative to Uvicorn for production deployments
"""

import os
import logging
from waitress import serve
from app.main import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_wsgi_app():
    """Create WSGI application for Waitress"""
    return app

def run_waitress_server():
    """Run the application using Waitress WSGI server"""
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    threads = int(os.getenv("WAITRESS_THREADS", 6))
    connection_limit = int(os.getenv("WAITRESS_CONNECTION_LIMIT", 1000))
    cleanup_interval = int(os.getenv("WAITRESS_CLEANUP_INTERVAL", 30))
    channel_timeout = int(os.getenv("WAITRESS_CHANNEL_TIMEOUT", 120))
    
    # Environment info
    environment = os.getenv("ENVIRONMENT", "production")
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info("🚀 Starting FTT-ML Backend with Waitress WSGI Server")
    logger.info(f"📍 Server: http://{host}:{port}")
    logger.info(f"🌍 Environment: {environment}")
    logger.info(f"🔧 Debug mode: {debug}")
    logger.info(f"🧵 Threads: {threads}")
    logger.info(f"🔗 Connection limit: {connection_limit}")
    logger.info(f"⏰ Channel timeout: {channel_timeout}s")
    logger.info(f"🧹 Cleanup interval: {cleanup_interval}s")
    
    try:
        # Start Waitress server
        serve(
            app,
            host=host,
            port=port,
            threads=threads,
            connection_limit=connection_limit,
            cleanup_interval=cleanup_interval,
            channel_timeout=channel_timeout,
            # Performance settings
            recv_bytes=65536,
            send_bytes=65536,
            # Security settings
            expose_tracebacks=debug,
            # Logging
            ident='ftt-ml-backend'
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {str(e)}")
        raise

if __name__ == "__main__":
    run_waitress_server()
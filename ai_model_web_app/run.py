"""
Script to run the FastAPI application
"""
import uvicorn
import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    print(f"""
    ╔═══════════════════════════════════════════════════╗
    ║   Air Quality AI - API Server                     ║
    ╠═══════════════════════════════════════════════════╣
    ║   Server: http://{host}:{port}                    ║
    ║   Docs:   http://{host}:{port}/docs               ║
    ║   Health: http://{host}:{port}/health             ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=debug,
        app_dir="src"
    )

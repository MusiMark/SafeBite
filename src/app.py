from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Model Web API",
    description="API for air quality prediction and anomaly detection",
    version="1.0.0"
)

# CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.on_event("startup")
async def startup_event():
    """Startup event - log that server is ready"""
    logger.info("=" * 60)
    logger.info("🚀 SafeBite API Server Started Successfully!")
    logger.info(f"📁 Static files directory: {static_path}")
    logger.info(f"🌐 Server is ready to accept connections")
    logger.info("=" * 60)

@app.get("/")
def read_root():
    """Serve the main HTML page"""
    return FileResponse(os.path.join(static_path, "index.html"))

@app.get("/dashboard.html")
def get_dashboard():
    """Serve the restaurant owner dashboard page"""
    return FileResponse(os.path.join(static_path, "dashboard.html"))

@app.get("/customer-app.html")
def get_customer_app():
    """Serve the customer app (leaderboard) page"""
    return FileResponse(os.path.join(static_path, "customer-app.html"))

@app.get("/user.html")
def get_user():
    """Serve the restaurant public profile page"""
    return FileResponse(os.path.join(static_path, "user.html"))

@app.get("/analytics.html")
def get_analytics():
    """Serve the analytics dashboard page"""
    return FileResponse(os.path.join(static_path, "analytics.html"))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

# Include routers - import here to avoid loading heavy ML libraries at startup
# The routers use lazy loading for ML models
from src.routers.inference import router
app.include_router(router, prefix="/api", tags=["inference"])

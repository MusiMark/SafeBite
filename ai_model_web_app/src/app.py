from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os


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

@app.get("/")
def read_root():
    """Serve the main HTML page"""
    return FileResponse(os.path.join(static_path, "index.html"))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

# Include routers - import here to avoid loading heavy ML libraries at startup
from src.routers.inference import router
app.include_router(router, prefix="/api", tags=["inference"])

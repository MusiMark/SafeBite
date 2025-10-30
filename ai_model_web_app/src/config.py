"""
Configuration management for the application
"""
import os
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Model Paths
    ANOMALY_MODEL_PATH: str = "../anomaly_detector/out_train"
    INSIDE_AIR_MODEL_PATH: str = "../inside_air_predictor/best_model.pth"
    EATSAFE_MODEL_PATH: str = "../safeBite_score/eatsafe_model.keras"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

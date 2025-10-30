@echo off
echo ================================================
echo   Air Quality AI - API Server
echo ================================================
echo.
echo Starting server...
echo.
echo Web UI:  http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo Health:  http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo ================================================
echo.

cd /d "%~dp0"
python run.py

@echo off
echo ========================================
echo RESTARTING AGENTBOOK BACKEND
echo ========================================
echo.

echo [1/3] Stopping existing backend...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *uvicorn*" 2>nul
timeout /t 2 /nobreak >nul

echo [2/3] Starting backend with new features...
cd /d D:\GenAI\DoAn01\backend
start "AgentBook Backend" cmd /k "uvicorn src.main:app --reload --host 0.0.0.0 --port 8000"

echo [3/3] Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo BACKEND RESTARTED SUCCESSFULLY!
echo ========================================
echo.
echo Backend URL: http://127.0.0.1:8000
echo API Docs:    http://127.0.0.1:8000/docs
echo.
echo New feature: Reasoning Path Visualization
echo - Check /api/v1/query/ask response schema
echo - Should have "reasoning_path" field
echo.
pause

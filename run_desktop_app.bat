@echo off
title Multi-Feature AI Content Moderation Desktop Container
cd /d "%~dp0"

echo =========================================================================
echo  Launching Multi-Feature AI Content Moderation Container...
echo  Loading latest model from models/final_model.keras
echo =========================================================================

python app_container.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Info] Webview container failed, falling back to Native Desktop GUI...
    python desktop_app.py
)

pause

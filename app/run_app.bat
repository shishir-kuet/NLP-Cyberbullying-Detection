@echo off
rem Double-click to start the demo. Close this window (or press Ctrl+C) to stop it.
rem Add --share after app.py for a temporary public link.
cd /d "%~dp0"
echo Starting Bangla Cyberbullying Classifier... (first start can take a minute)
.venv\Scripts\python app.py %*
pause

@echo off
echo Starting OOTP Perfect Team Optimizer...
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please install Python 3.11+
    pause
    exit /b 1
)

REM First run setup
if not exist "data\ootp_optimizer.db" (
    echo First run detected - setting up database...
    python setup.py
)

REM Install/update dependencies
pip install -r requirements.txt -q

REM Start file watcher in background
start /B python -m app.core.file_watcher

REM Start Streamlit dashboard
streamlit run app/main.py --server.port 8501 --theme.base dark --server.headless true

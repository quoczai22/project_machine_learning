@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Dang khoi dong demo Telco Churn tai http://localhost:8501 ...
python -m streamlit run app.py --server.port 8501 --server.headless false
pause

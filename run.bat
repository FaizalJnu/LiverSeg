@echo off

echo Starting script 1...
python transunet_train.py
:: Check if the script failed (Optional)
if %errorlevel% neq 0 exit /b %errorlevel%

echo Starting script 2...
python inference_transunet.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo Starting script 3...
python sample_results.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo All scripts finished.
pause
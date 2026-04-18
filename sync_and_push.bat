@echo off
echo [1/3] Syncing with GitHub (Pulling latest changes)...
git pull origin main --rebase
if %ERRORLEVEL% NEQ 0 (
    echo Error: Could not sync with GitHub. Check your internet connection.
    pause
    exit /b %ERRORLEVEL%
)

echo [2/3] Adding new requirements and fixes...
git add .
git commit -m "Fix: Integrated AI engine and updated requirements.txt"

echo [3/3] Pushing final version to GitHub...
git push origin main
if %ERRORLEVEL% NEQ 0 (
    echo Error: Push failed.
    pause
    exit /b %ERRORLEVEL%
)

echo Done! Your site on Streamlit Cloud will now update automatically.
pause

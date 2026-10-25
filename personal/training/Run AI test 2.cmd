@echo off
title Spire Mentor AI test (safe route)
rem Runs the four-way run-AI simulator test (abtests\ab3.sh). It waits until
rem Slay the Spire 2 has been closed for 15 minutes, backs off if you start
rem playing, and starts AI training again when it finishes.
"C:\Program Files\Git\bin\bash.exe" "%~dp0abtests\ab3.sh"
echo.
echo AI test finished or stopped. Results: %~dp0abtests\run3\result.txt
pause

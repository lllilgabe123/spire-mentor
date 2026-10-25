@echo off
title Spire Mentor AI test
rem Same as "Run AI test.cmd" but starts at once instead of waiting for the
rem game to stay closed 15 minutes. Still backs off if you start the game.
set SKIP_WAIT=1
"C:\Program Files\Git\bin\bash.exe" "%~dp0abtests\ab.sh"
echo.
echo AI test finished or stopped. Results: %~dp0abtests\run2\result.txt
pause

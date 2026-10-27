

@echo on

pip check
IF %ERRORLEVEL% NEQ 0 exit /B 1
python -m conda info | grep -q "user-agent.*aau/"
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0

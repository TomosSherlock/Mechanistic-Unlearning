

@echo on

pytest -k test_index_on_single_subdir_1
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0

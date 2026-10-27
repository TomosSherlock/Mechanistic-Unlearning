



conda-token --help
IF %ERRORLEVEL% NEQ 0 exit /B 1
anaconda auth --help
IF %ERRORLEVEL% NEQ 0 exit /B 1
python -c "from anaconda_auth import __version__; assert __version__ == \"0.12.3\""
IF %ERRORLEVEL% NEQ 0 exit /B 1
python -m anaconda_auth._conda.config --verify
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0

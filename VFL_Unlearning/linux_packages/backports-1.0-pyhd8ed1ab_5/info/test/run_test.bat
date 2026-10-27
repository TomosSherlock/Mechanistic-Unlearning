



unset PIP_NO_INDEX
IF %ERRORLEVEL% NEQ 0 exit /B 1
python -m pip install --target alt-site backports.tarfile
IF %ERRORLEVEL% NEQ 0 exit /B 1
export PYTHONPATH="${SP_DIR}:alt-site"
IF %ERRORLEVEL% NEQ 0 exit /B 1
python -c "import backports.tarfile"
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0

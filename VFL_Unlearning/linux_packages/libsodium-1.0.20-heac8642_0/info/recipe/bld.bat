@echo on

:: Pick the correct architecture
if "%ARCH%" == "32" (
  set ARCH=Win32
) else (
  set ARCH=x64
)

:: Set temp folder for the outputs
set ARTIFACTS_DIR=%LIBRARY_PREFIX%\TEMP
cd /d %SRC_DIR%\builds\msvc\vs%VS_YEAR%\
msbuild libsodium.sln /p:Configuration=DynRelease /p:Platform=%ARCH% /p:OutDir=%ARTIFACTS_DIR%\
if errorlevel 1 exit 1

:: Check for built DLLs
if not exist %ARTIFACTS_DIR%\libsodium.dll    exit 1

:: Copy dynamic lib, import lib and headers to their correct destinations
move /y %ARTIFACTS_DIR%\libsodium.dll %LIBRARY_BIN%
move /y  %ARTIFACTS_DIR%\libsodium.lib %LIBRARY_LIB%
xcopy /s /y /i %SRC_DIR%\src\libsodium\include\sodium %LIBRARY_INC%\sodium
xcopy /s /y %SRC_DIR%\src\libsodium\include\sodium.h %LIBRARY_INC%\

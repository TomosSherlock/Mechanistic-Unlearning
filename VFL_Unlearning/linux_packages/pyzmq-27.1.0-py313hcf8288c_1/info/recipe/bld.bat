set PYZMQ_NO_BUNDLE=1
set SKBUILD_CMAKE_VERBOSE=true
set CMAKE_BUILD_TYPE=Release

%PYTHON% -m pip install --no-deps --no-build-isolation -vv .
if errorlevel 1 exit 1
set PYZMQ_NO_BUNDLE=1
set SKBUILD_CMAKE_VERBOSE=true
set CMAKE_GENERATOR=Ninja

"%PYTHON%" -m pip install --no-deps --no-build-isolation --config-settings="cmake.build-type=Release" -vv .
if errorlevel 1 exit 1

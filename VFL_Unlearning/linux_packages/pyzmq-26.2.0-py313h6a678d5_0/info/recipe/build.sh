set -x

# make sure it doesn't fallback on bundled libzmq
export PYZMQ_NO_BUNDLE=1
export SKBUILD_CMAKE_VERBOSE=true

"${PYTHON}" -m pip install --no-deps --no-build-isolation .

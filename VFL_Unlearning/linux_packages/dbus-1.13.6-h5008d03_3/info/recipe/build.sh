#!/bin/bash
# Get an updated config.sub and config.guess
cp $BUILD_PREFIX/share/libtool/build-aux/config.* ./build-aux

CPPFLAGS="${CPPFLAGS} -I${PREFIX}/include" \
LDFLAGS="${LDFLAGS} -L${PREFIX}/lib -Wl,-rpath,${PREFIX}/lib"  \
  ./configure --prefix=${PREFIX}   \
              --disable-systemd    \
              --disable-selinux    \
              --disable-xml-docs   \
              --with-launchd-agent-dir=${PREFIX}  \
              --without-x

make -j${CPU_COUNT} ${VERBOSE_AT}
if [[ $(uname) != Darwin ]]; then
  make check
fi
make install
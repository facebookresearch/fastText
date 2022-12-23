#!/bin/bash
#
# Called inside the manylinux image
#
# based on https://github.com/lxml/lxml/blob/b224e0f69dde58425d1077e07d193d19d3f803a9/tools/manylinux/build-wheels.sh
echo "Started $0 $@"

set -e -x
[ -n "$WHEELHOUSE" ] || WHEELHOUSE=wheelhouse
SDIST=$1
PACKAGE=$(basename ${SDIST%-*})
SDIST_PREFIX='fasttext_predict'
[ -z "$PYTHON_BUILD_VERSION" ] && PYTHON_BUILD_VERSION="*"

build_wheel() {
    echo "===[ build_wheels $1 $2 ]==="
    pybin="$1"
    source="$2"
    [ -n "$source" ] || source=/io

    ${pybin}/pip install --upgrade pip

    case $( uname -m ) in
        x86_64|i686|amd64) CFLAGS="$CFLAGS -march=core2";;
        aarch64) CFLAGS="$CFLAGS -march=armv8-a -mtune=cortex-a72";;
    esac

    rm -rf /io/build
    env STATIC_DEPS=true \
        RUN_TESTS=true \
        LDFLAGS="$LDFLAGS -fPIC" \
        CFLAGS="$CFLAGS -fPIC" \
        ACLOCAL_PATH=/usr/share/aclocal/ \
        ${pybin}/pip \
            wheel \
            -v \
            "$source" \
            -w /io/$WHEELHOUSE
}

prepare_system() {
    echo "===[ prepare_system ]==="
    rm -fr /opt/python/cp27-*
    rm -fr /opt/python/cp34-*
    rm -fr /opt/python/cp35-*
    rm -fr /opt/python/cp36-*
    echo "Python versions found: $(cd /opt/python && echo cp* | sed -e 's|[^ ]*-||g')"
    ${CC:-gcc} --version
}

build_wheels() {
    echo "===[ build_wheels ]==="
    # Compile wheels for all python versions
    test -e "$SDIST" && source="$SDIST" || source=
    for PYBIN in /opt/python/${PYTHON_BUILD_VERSION}/bin; do
        echo "Starting build with $($PYBIN/python -V)"
        build_wheel "$PYBIN" "$source"
    done
}

repair_wheels() {
    echo "===[ repair_wheels ]==="
    # Bundle external shared libraries into the wheels
    for whl in /io/$WHEELHOUSE/${SDIST_PREFIX}-*.whl; do
        OPT="--strip"
        if [[ "$whl" == *x86_64.whl && "$whl" == *manylinux_2_24_x86_64* ]]; then
            OPT="$OPT --plat manylinux_2_34_x86_64"
        fi
        if [[ "$whl" == *x86_64.whl && "$whl" == *manylinux1_x86_64* ]]; then
            OPT="$OPT --plat manylinux1_x86_64"
        fi
        auditwheel show $whl
        auditwheel repair $whl $OPT -w /io/$WHEELHOUSE || exit 1
    done
}

show_wheels() {
    echo "===[ show_wheels ]==="
    ls -l /io/$WHEELHOUSE/${SDIST_PREFIX}-*.whl
}

prepare_system
build_wheels
repair_wheels
show_wheels

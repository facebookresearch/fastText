#!/bin/bash
#
# Called inside the manylinux image
#
# based on https://github.com/lxml/lxml/blob/b224e0f69dde58425d1077e07d193d19d3f803a9/tools/manylinux/build-wheels.sh
echo "Started $0 $@"

set -e -x
REQUIREMENTS=/io/requirements.txt
[ -n "$WHEELHOUSE" ] || WHEELHOUSE=wheelhouse
SDIST=$1
PACKAGE=$(basename ${SDIST%-*})
SDIST_PREFIX='fasttext_predict'
[ -z "$PYTHON_BUILD_VERSION" ] && PYTHON_BUILD_VERSION="*"

build_wheel() {
    pybin="$1"
    source="$2"
    [ -n "$source" ] || source=/io

    env STATIC_DEPS=true \
        RUN_TESTS=true \
        LDFLAGS="$LDFLAGS -fPIC" \
        CFLAGS="$CFLAGS -fPIC" \
        ACLOCAL_PATH=/usr/share/aclocal/ \
        ${pybin}/pip \
            wheel \
            "$source" \
            -w /io/$WHEELHOUSE
}

prepare_system() {
    rm -fr /opt/python/cp27-*
    rm -fr /opt/python/cp34-*
    rm -fr /opt/python/cp35-*
    rm -fr /opt/python/cp36-*
    echo "Python versions found: $(cd /opt/python && echo cp* | sed -e 's|[^ ]*-||g')"
    ${CC:-gcc} --version
}

build_wheels() {
    # Compile wheels for all python versions
    test -e "$SDIST" && source="$SDIST" || source=
    FIRST=
    SECOND=
    THIRD=
    for PYBIN in /opt/python/${PYTHON_BUILD_VERSION}/bin; do
        # Install build requirements if we need them and file exists
        test -n "$source" -o ! -e "$REQUIREMENTS" \
            || ${PYBIN}/python -m pip install -r "$REQUIREMENTS"

        echo "Starting build with $($PYBIN/python -V)"
        build_wheel "$PYBIN" "$source" &
        THIRD=$!

        [ -z "$FIRST" ] || wait ${FIRST}
        if [ "$(uname -m)" == "aarch64" ]; then FIRST=$THIRD; else FIRST=$SECOND; fi
        SECOND=$THIRD
    done
    wait || exit 1
}

repair_wheels() {
    # Bundle external shared libraries into the wheels
    find /io/$WHEELHOUSE
    find /usr/lib* | grep libm
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
    ls -l /io/$WHEELHOUSE/${SDIST_PREFIX}-*.whl
}

prepare_system
build_wheels
repair_wheels
show_wheels

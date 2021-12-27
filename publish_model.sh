#!/bin/bash

TESTPYPI_USERNAME=$TWINE_USERNAME
TESTPYPI_PSW=$TWINE_PASSWORD

set -e

DIRS="$@"
BASE_DIR=$(pwd)
SETUP="setup.py"

warn() {
    echo "$@" 1>&2
}

die() {
    warn "$@"
    exit 1
}

build() {
    DIR="${1/%\//}"
    echo "Checking directory $DIR"
    cd "$BASE_DIR/$DIR"
    [ ! -e $SETUP ] && warn "No $SETUP file, skipping" && return
    PACKAGE_NAME=$(python $SETUP --fullname)
    echo "Package $PACKAGE_NAME"
    python "$SETUP" sdist bdist_wheel || die "Building package $PACKAGE_NAME failed"
    pip install twine
    twine upload -r testpypi -u "$TESTPYPI_USERNAME" -p "$TESTPYPI_PSW" dist/*
}

if [ -n "$DIRS" ]; then
    for dir in $DIRS; do
        build $dir
    done
else
    ls -d */ | while read dir; do
        build $dir
    done
fi
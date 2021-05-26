#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

echo "Building match_class op..."
cd ops/match_class
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

PYTHON=${PYTHON:-"python"}

echo "Building match_boundary op..."
cd ../match_boundary
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

PYTHON=${PYTHON:-"python"}

echo "Building follow_cluster op..."
cd ../follow_cluster
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

PYTHON=${PYTHON:-"python"}

echo "Building vcount_cluster op..."
cd ../vcount_cluster
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

PYTHON=${PYTHON:-"python"}

echo "Building split_repscore op..."
cd ../split_repscore
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

PYTHON=${PYTHON:-"python"}

echo "Building intra_collection op..."
cd ../intra_collection
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace
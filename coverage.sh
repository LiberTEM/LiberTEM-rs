#!/bin/bash

set -eu

# oof...
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

# pre-flight check: need to have a virtual environment active with numpy installed:
python -c 'import numpy'

cargo tarpaulin --engine llvm -o html

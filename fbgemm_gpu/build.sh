#!/bin/bash

export MAX_JOBS=32
python3.6 setup.py build develop 2>&1 | tee build.log

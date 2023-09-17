#!/bin/bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install pupil-detectors pye3d mediapipe poetry python-osc opencv-python scipy numba
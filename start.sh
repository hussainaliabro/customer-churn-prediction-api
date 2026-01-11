#!/usr/bin/env bash
set -e

echo "Training model..."
python -m ml.train

echo "Starting FastAPI..."
uvicorn app.main:app --host 0.0.0.0 --port 8000

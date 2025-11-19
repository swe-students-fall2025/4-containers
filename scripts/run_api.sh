#!/usr/bin/env bash
set -euo pipefail
cd web-app
export FLASK_APP=app.py
flask run --host 0.0.0.0 --port 8000
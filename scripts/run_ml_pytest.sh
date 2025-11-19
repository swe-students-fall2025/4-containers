#!/usr/bin/env bash
set -euo pipefail
cd machine-learning-client
pipenv run pytest -q -m "not slow"
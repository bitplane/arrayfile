#!/usr/bin/env bash

source .venv/bin/activate

# Skip first parameter, pass rest to pytest
shift
pytest "$@"

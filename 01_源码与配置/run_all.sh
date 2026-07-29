#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" != "--verify-existing" ]; then
  echo "Usage: bash 01_源码与配置/run_all.sh --verify-existing"
  echo "This verification entrypoint never calls the DeepSeek API."
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHONPATH='01_源码与配置' /opt/anaconda3/bin/python -m mark10.audit
PYTHONPATH='01_源码与配置' /opt/anaconda3/bin/python -m pytest tests -q

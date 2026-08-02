#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "Running sbi_NLE.py..."
python sbi_NLE.py

echo "Running sbi_NRE.py..."
python sbi_NRE.py

echo "Done."

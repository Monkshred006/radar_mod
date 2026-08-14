#!/usr/bin/env bash
set -e

echo "================================================================"
echo " PhotonShield AI — Downloading & Ingesting RaDICaL Dataset"
echo "================================================================"

python scripts/download_radical.py --output-dir data/radical --total-samples 500

echo "RaDICaL dataset download and directory structure verified."

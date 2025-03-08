#!/bin/bash
apt-get update && apt-get install -y build-essential python3-dev
pip install --no-cache-dir -r requirements.txt

#!/bin/bash
set -e

echo "Starting JupyterHub with dynamic user creation..."

# Create users from userlist.txt
/usr/local/bin/create-users.sh

# Start JupyterHub
echo "Starting JupyterHub..."
exec jupyterhub --ip=0.0.0.0 --port=8000 --config=/etc/jupyterhub/jupyterhub_config.py "$@"
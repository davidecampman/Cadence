#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND_DIR="$REPO_ROOT/frontend"

# Rebuild frontend if dependencies or source changed
if [ -n "$(git diff HEAD@{1} --name-only -- frontend/package-lock.json 2>/dev/null)" ]; then
    echo ">> package-lock.json changed — reinstalling dependencies..."
    (cd "$FRONTEND_DIR" && npm ci)
elif [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo ">> node_modules missing — installing dependencies..."
    (cd "$FRONTEND_DIR" && npm install)
fi

if [ -n "$(git diff HEAD@{1} --name-only -- frontend/ 2>/dev/null)" ]; then
    echo ">> Frontend files changed — rebuilding..."
    (cd "$FRONTEND_DIR" && npm run build)
    echo ">> Frontend build complete."
else
    echo ">> No frontend changes detected — skipping build."
fi

# Install Python dependencies if pyproject.toml changed
if [ -n "$(git diff HEAD@{1} --name-only -- pyproject.toml 2>/dev/null)" ]; then
    echo ">> pyproject.toml changed — reinstalling Python dependencies..."
    pip install -e "$REPO_ROOT" --quiet
fi

# Start the server
echo ">> Starting Sentinel server..."
exec python -m sentinel.server "$@"

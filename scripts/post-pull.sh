#!/usr/bin/env bash
set -euo pipefail

# Post-pull hook: refreshes dependencies and rebuilds as needed after git pull.
# Can also be run standalone. For full setup from scratch, use ./start.sh instead.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND_DIR="$REPO_ROOT/frontend"
VENV_DIR="$REPO_ROOT/.venv"

info()  { echo -e "\033[1;34m>>\033[0m $*"; }
ok()    { echo -e "\033[1;32m>>\033[0m $*"; }

# ---------- Python venv ----------
if [ -d "$VENV_DIR" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# ---------- Python dependencies ----------
if [ -n "$(git diff HEAD@{1} --name-only -- pyproject.toml 2>/dev/null)" ]; then
    info "pyproject.toml changed — reinstalling Python dependencies..."
    pip install -e "$REPO_ROOT" --quiet
    ok "Python dependencies updated."
fi

# ---------- Frontend dependencies ----------
if [ -n "$(git diff HEAD@{1} --name-only -- frontend/package-lock.json 2>/dev/null)" ]; then
    info "package-lock.json changed — reinstalling frontend dependencies..."
    (cd "$FRONTEND_DIR" && npm ci)
elif [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    info "node_modules missing — installing frontend dependencies..."
    (cd "$FRONTEND_DIR" && npm install)
fi

# ---------- Frontend build ----------
if [ -n "$(git diff HEAD@{1} --name-only -- frontend/ 2>/dev/null)" ]; then
    info "Frontend files changed — rebuilding..."
    (cd "$FRONTEND_DIR" && npm run build)
    ok "Frontend rebuilt."
else
    ok "No frontend changes — skipping build."
fi

# ---------- Restart server ----------
if command -v lsof &>/dev/null && lsof -ti :8000 &>/dev/null; then
    info "Port 8000 is in use — stopping existing process..."
    lsof -ti :8000 | xargs kill 2>/dev/null || true
    sleep 1
fi

ok "Post-pull complete — starting Cadence server..."
exec python -m cadence.server "$@"

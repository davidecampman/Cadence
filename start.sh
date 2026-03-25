#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$REPO_ROOT/frontend"
VENV_DIR="$REPO_ROOT/.venv"
MIN_PYTHON="3.11"
MIN_NODE="18"

# ---------- helpers ----------
info()  { echo -e "\033[1;34m>>\033[0m $*"; }
ok()    { echo -e "\033[1;32m>>\033[0m $*"; }
err()   { echo -e "\033[1;31m>>\033[0m $*" >&2; }

version_ge() {
    # returns 0 if $1 >= $2 (dotted version comparison)
    printf '%s\n%s' "$2" "$1" | sort -t. -k1,1n -k2,2n -k3,3n -C
}

# ---------- Python ----------
info "Checking Python..."
PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
        if [ -n "$ver" ] && version_ge "$ver" "$MIN_PYTHON"; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    err "Python >= $MIN_PYTHON is required but was not found."
    err "Install it from https://www.python.org/downloads/ and re-run this script."
    exit 1
fi
ok "Found $PYTHON ($ver)"

# ---------- venv ----------
if [ ! -d "$VENV_DIR" ]; then
    info "Creating Python virtual environment in .venv..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Virtual environment created."
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Activated venv: $(which python)"

# ---------- Python dependencies ----------
# Always ensure deps are up-to-date (pip is fast when nothing changed)
info "Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -e "$REPO_ROOT" --quiet
ok "Python dependencies installed."

# ---------- Node / npm ----------
info "Checking Node.js..."
if ! command -v node &>/dev/null; then
    err "Node.js >= $MIN_NODE is required but was not found."
    err "Install it from https://nodejs.org/ and re-run this script."
    exit 1
fi

node_ver=$(node -v | sed 's/^v//')
node_major=${node_ver%%.*}
if [ "$node_major" -lt "$MIN_NODE" ]; then
    err "Node.js >= $MIN_NODE is required (found $node_ver)."
    exit 1
fi
ok "Found Node.js $node_ver"

# ---------- Frontend dependencies ----------
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    info "Installing frontend dependencies..."
    (cd "$FRONTEND_DIR" && npm install)
    ok "Frontend dependencies installed."
else
    # If package-lock.json is newer than node_modules, reinstall
    if [ "$FRONTEND_DIR/package-lock.json" -nt "$FRONTEND_DIR/node_modules/.package-lock.json" ] 2>/dev/null; then
        info "package-lock.json changed — reinstalling frontend dependencies..."
        (cd "$FRONTEND_DIR" && npm ci)
        ok "Frontend dependencies reinstalled."
    else
        ok "Frontend dependencies up to date."
    fi
fi

# ---------- Frontend build ----------
if [ ! -d "$FRONTEND_DIR/dist" ]; then
    info "Building frontend..."
    (cd "$FRONTEND_DIR" && npm run build)
    ok "Frontend built."
else
    # Rebuild if any source file is newer than dist
    newest_src=$(find "$FRONTEND_DIR/src" "$FRONTEND_DIR/index.html" "$FRONTEND_DIR/tsconfig"*.json -maxdepth 2 -newer "$FRONTEND_DIR/dist" 2>/dev/null | head -1 || true)
    if [ -n "$newest_src" ]; then
        info "Frontend sources changed — rebuilding..."
        (cd "$FRONTEND_DIR" && npm run build)
        ok "Frontend rebuilt."
    else
        ok "Frontend build up to date."
    fi
fi

# ---------- Git hooks ----------
if [ -d "$REPO_ROOT/.git" ]; then
    git -C "$REPO_ROOT" config core.hooksPath .githooks 2>/dev/null || true
fi

# ---------- Kill existing process on port 8000 ----------
if command -v lsof &>/dev/null && lsof -ti :8000 &>/dev/null; then
    info "Port 8000 is in use — stopping existing process..."
    lsof -ti :8000 | xargs kill 2>/dev/null || true
    sleep 1
fi

# ---------- Start ----------
ok "Setup complete — starting Cadence server..."
exec python -m cadence.server "$@"

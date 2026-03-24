#!/usr/bin/env bash
# bundle-python.sh — Download standalone Python and install Sentinel into it.
#
# This creates a self-contained Python environment with all dependencies
# that gets bundled into the Tauri app as a resource.
#
# Usage:
#   ./scripts/bundle-python.sh [target]
#
# Targets: macos-aarch64, macos-x86_64, windows-x86_64, linux-x86_64
# Default: auto-detect current platform.

set -euo pipefail

PYTHON_VERSION="3.11.9"
STANDALONE_BASE_URL="https://github.com/indygreg/python-build-standalone/releases/download/20240415"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/src-tauri/python-runtime"

# --- Detect or accept target ---
detect_target() {
    local os arch
    os="$(uname -s | tr '[:upper:]' '[:lower:]')"
    arch="$(uname -m)"

    case "${os}" in
        darwin) os="macos" ;;
        linux)  os="linux" ;;
        mingw*|msys*|cygwin*) os="windows" ;;
    esac

    case "${arch}" in
        x86_64|amd64) arch="x86_64" ;;
        arm64|aarch64) arch="aarch64" ;;
    esac

    echo "${os}-${arch}"
}

TARGET="${1:-$(detect_target)}"

echo "==> Bundling Python ${PYTHON_VERSION} for ${TARGET}"
echo "    Output: ${OUTPUT_DIR}"

# --- Map target to download filename ---
case "${TARGET}" in
    macos-aarch64)
        ARCHIVE="cpython-${PYTHON_VERSION}+20240415-aarch64-apple-darwin-install_only_stripped.tar.gz"
        ;;
    macos-x86_64)
        ARCHIVE="cpython-${PYTHON_VERSION}+20240415-x86_64-apple-darwin-install_only_stripped.tar.gz"
        ;;
    windows-x86_64)
        ARCHIVE="cpython-${PYTHON_VERSION}+20240415-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"
        ;;
    linux-x86_64)
        ARCHIVE="cpython-${PYTHON_VERSION}+20240415-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
        ;;
    *)
        echo "ERROR: Unsupported target: ${TARGET}" >&2
        exit 1
        ;;
esac

DOWNLOAD_URL="${STANDALONE_BASE_URL}/${ARCHIVE}"

# --- Download ---
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT

echo "==> Downloading ${DOWNLOAD_URL}"
curl -fSL --retry 3 -o "${TEMP_DIR}/${ARCHIVE}" "${DOWNLOAD_URL}"

# --- Extract ---
echo "==> Extracting Python runtime"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
tar -xzf "${TEMP_DIR}/${ARCHIVE}" -C "${OUTPUT_DIR}" --strip-components=1

# --- Install Sentinel into the bundled Python ---
BUNDLED_PIP="${OUTPUT_DIR}/bin/pip3"
if [[ "${TARGET}" == windows-* ]]; then
    BUNDLED_PIP="${OUTPUT_DIR}/python.exe -m pip"
fi

echo "==> Installing Sentinel into bundled Python"
${BUNDLED_PIP} install --no-cache-dir "${PROJECT_ROOT}"

# --- Trim unnecessary files to reduce bundle size ---
echo "==> Trimming bundle"
rm -rf "${OUTPUT_DIR}/share"
rm -rf "${OUTPUT_DIR}/lib/python3.11/test"
rm -rf "${OUTPUT_DIR}/lib/python3.11/unittest"
rm -rf "${OUTPUT_DIR}/lib/python3.11/idlelib"
rm -rf "${OUTPUT_DIR}/lib/python3.11/tkinter"
rm -rf "${OUTPUT_DIR}/lib/python3.11/turtle*"
rm -rf "${OUTPUT_DIR}/lib/python3.11/ensurepip"
find "${OUTPUT_DIR}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "${OUTPUT_DIR}" -name "*.pyc" -delete 2>/dev/null || true

# --- Report size ---
BUNDLE_SIZE=$(du -sh "${OUTPUT_DIR}" | cut -f1)
echo "==> Done! Bundled Python size: ${BUNDLE_SIZE}"
echo "    Ready for: tauri build"

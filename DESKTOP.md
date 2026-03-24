# Sentinel Desktop

Sentinel can be packaged as a native desktop application for macOS, Windows, and Linux using [Tauri](https://tauri.app).

The desktop app wraps the existing React frontend in a native window and manages the Python backend as a sidecar process.

## Architecture

```
Tauri App
├── Native window (WebView)
│   └── React frontend (bundled static files)
├── Sidecar: Python runtime + sentinel-server
│   ├── Standalone CPython 3.11 (bundled)
│   ├── All pip dependencies pre-installed
│   └── Runs on localhost:8000
└── Rust core
    ├── Process lifecycle management
    ├── System tray icon
    └── Auto-updater (future)
```

## Prerequisites

- **Rust** (latest stable): https://rustup.rs
- **Node.js** 20+
- **Python** 3.11+ (for development builds only — production bundles its own)

### Platform-specific

**macOS:**
- Xcode Command Line Tools: `xcode-select --install`

**Windows:**
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with "Desktop development with C++"
- [WebView2](https://developer.microsoft.com/en-us/microsoft-edge/webview2/) (pre-installed on Windows 10 1803+ and Windows 11)

**Linux:**
```bash
sudo apt install libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf
```

## Development

```bash
# Install root dependencies (Tauri CLI)
npm install

# Install frontend dependencies
npm run frontend:install

# Install Python package in dev mode
pip install -e ".[dev]"

# Start Tauri dev mode (hot-reload for frontend, uses system Python)
npm run tauri:dev
```

In dev mode:
- The React frontend runs on `localhost:5173` with hot-reload
- You need to start the Python backend separately: `sentinel-server`
- Tauri opens a native window pointing at the Vite dev server

## Production Build

```bash
# 1. Bundle standalone Python with all dependencies
npm run tauri:bundle-python

# 2. Build the desktop app
npm run tauri:build
```

Build outputs:
- **macOS:** `src-tauri/target/release/bundle/dmg/Sentinel.dmg`
- **Windows:** `src-tauri/target/release/bundle/nsis/Sentinel_Setup.exe`
- **Linux:** `src-tauri/target/release/bundle/deb/sentinel.deb`

## CI/CD

The GitHub Actions workflow at `.github/workflows/desktop-build.yml` builds for all platforms:
- Triggered on version tags (`v*`) or manual dispatch
- Builds `.dmg`, `.msi`/`.exe`, `.deb`, and `.AppImage`
- Uploads artifacts and creates draft GitHub Releases

### Code Signing

Set these secrets in GitHub for signed builds:

**macOS:**
- `APPLE_CERTIFICATE` — Base64-encoded .p12 certificate
- `APPLE_CERTIFICATE_PASSWORD`
- `APPLE_SIGNING_IDENTITY` — e.g., "Developer ID Application: Your Name (TEAMID)"
- `APPLE_ID`, `APPLE_PASSWORD`, `APPLE_TEAM_ID` — for notarization

**Windows:**
- Configure `certificateThumbprint` in `src-tauri/tauri.conf.json`

## How It Works

1. **App launch:** Tauri starts, resolves the bundled Python runtime from its resources directory
2. **Sidecar start:** Spawns `python3 -m sentinel.server --host 127.0.0.1 --port 8000`
3. **Frontend load:** The React app detects it's in desktop mode (`__TAURI_INTERNALS__`) and routes API calls to `http://localhost:8000` instead of using relative paths
4. **Shutdown:** When the window closes, Tauri kills the Python sidecar process

## Configuration

The desktop app uses the same `config/default.yaml` and environment variables as the CLI/web versions. The `SENTINEL_DESKTOP=1` env var is set automatically so the backend can detect it's running inside the desktop app.

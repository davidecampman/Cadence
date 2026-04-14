<#
.SYNOPSIS
    Bootstrap and start the Cadence development server (Windows / PowerShell).
.DESCRIPTION
    PowerShell equivalent of start.sh — creates a Python venv, installs
    dependencies, builds the frontend, and launches the server.
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments)]
    [string[]]$ServerArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot    = Split-Path -Parent $MyInvocation.MyCommand.Definition
$FrontendDir = Join-Path $RepoRoot 'frontend'
$VenvDir     = Join-Path $RepoRoot '.venv'
$MinPython   = [version]'3.11'
$MinNode     = 18

# ---------- helpers ----------
function Info  { param([string]$msg) Write-Host ">> $msg" -ForegroundColor Blue }
function Ok    { param([string]$msg) Write-Host ">> $msg" -ForegroundColor Green }
function Err   { param([string]$msg) Write-Host ">> $msg" -ForegroundColor Red }

# ---------- Python ----------
Info 'Checking Python...'
$python = $null
foreach ($candidate in @('python3', 'python')) {
    $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($cmd) {
        $verOutput = & $candidate --version 2>&1
        if ($verOutput -match '(\d+\.\d+)') {
            $verStr = $Matches[1]
            if ([version]$verStr -ge $MinPython) {
                $python = $candidate
                break
            }
        }
    }
}

if (-not $python) {
    Err "Python >= $MinPython is required but was not found."
    Err 'Install it from https://www.python.org/downloads/ and re-run this script.'
    exit 1
}
Ok "Found $python ($verStr)"

# ---------- venv ----------
if (-not (Test-Path $VenvDir)) {
    Info 'Creating Python virtual environment in .venv...'
    & $python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Err 'Failed to create virtual environment.'; exit 1 }
    Ok 'Virtual environment created.'
}

# Activate venv
$activateScript = Join-Path $VenvDir 'Scripts' 'Activate.ps1'
if (-not (Test-Path $activateScript)) {
    # Linux/macOS venv layout (in case someone runs pwsh on *nix)
    $activateScript = Join-Path $VenvDir 'bin' 'Activate.ps1'
}
if (-not (Test-Path $activateScript)) {
    Err "Cannot find venv activation script at $activateScript"
    exit 1
}
. $activateScript
Info "Activated venv: $(Get-Command python | Select-Object -ExpandProperty Source)"

# ---------- Python dependencies ----------
Info 'Installing Python dependencies...'
& python -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { Err 'pip upgrade failed.'; exit 1 }
& python -m pip install -e $RepoRoot --quiet
if ($LASTEXITCODE -ne 0) { Err 'Failed to install Python dependencies.'; exit 1 }
Ok 'Python dependencies installed.'

# ---------- Node / npm ----------
Info 'Checking Node.js...'
$nodeCmd = Get-Command node -ErrorAction SilentlyContinue
if (-not $nodeCmd) {
    Err "Node.js >= $MinNode is required but was not found."
    Err 'Install it from https://nodejs.org/ and re-run this script.'
    exit 1
}

$nodeVer = (node -v) -replace '^v', ''
$nodeMajor = [int]($nodeVer.Split('.')[0])
if ($nodeMajor -lt $MinNode) {
    Err "Node.js >= $MinNode is required (found $nodeVer)."
    exit 1
}
Ok "Found Node.js $nodeVer"

# ---------- Frontend dependencies ----------
$nodeModules = Join-Path $FrontendDir 'node_modules'
$packageLock = Join-Path $FrontendDir 'package-lock.json'
$packageLockInner = Join-Path $nodeModules '.package-lock.json'

if (-not (Test-Path $nodeModules)) {
    Info 'Installing frontend dependencies...'
    Push-Location $FrontendDir
    npm install
    if ($LASTEXITCODE -ne 0) { Pop-Location; Err 'npm install failed.'; exit 1 }
    Pop-Location
    Ok 'Frontend dependencies installed.'
} elseif ((Test-Path $packageLock) -and (Test-Path $packageLockInner) -and
          ((Get-Item $packageLock).LastWriteTime -gt (Get-Item $packageLockInner).LastWriteTime)) {
    Info 'package-lock.json changed — reinstalling frontend dependencies...'
    Push-Location $FrontendDir
    npm ci
    if ($LASTEXITCODE -ne 0) { Pop-Location; Err 'npm ci failed.'; exit 1 }
    Pop-Location
    Ok 'Frontend dependencies reinstalled.'
} else {
    Ok 'Frontend dependencies up to date.'
}

# ---------- Frontend build ----------
$distDir = Join-Path $FrontendDir 'dist'

if (-not (Test-Path $distDir)) {
    Info 'Building frontend...'
    Push-Location $FrontendDir
    npm run build
    if ($LASTEXITCODE -ne 0) { Pop-Location; Err 'Frontend build failed.'; exit 1 }
    Pop-Location
    Ok 'Frontend built.'
} else {
    # Rebuild if any source file is newer than dist
    $distTime = (Get-Item $distDir).LastWriteTime
    $srcDir   = Join-Path $FrontendDir 'src'
    $newerFiles = @(
        Get-ChildItem -Path $srcDir -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.LastWriteTime -gt $distTime }
    )
    # Also check index.html and tsconfig files
    foreach ($extra in @(
        (Join-Path $FrontendDir 'index.html'),
        (Join-Path $FrontendDir 'tsconfig.json'),
        (Join-Path $FrontendDir 'tsconfig.app.json'),
        (Join-Path $FrontendDir 'tsconfig.node.json')
    )) {
        if ((Test-Path $extra) -and (Get-Item $extra).LastWriteTime -gt $distTime) {
            $newerFiles += Get-Item $extra
        }
    }

    if ($newerFiles.Count -gt 0) {
        Info 'Frontend sources changed — rebuilding...'
        Push-Location $FrontendDir
        npm run build
        if ($LASTEXITCODE -ne 0) { Pop-Location; Err 'Frontend rebuild failed.'; exit 1 }
        Pop-Location
        Ok 'Frontend rebuilt.'
    } else {
        Ok 'Frontend build up to date.'
    }
}

# ---------- Git hooks ----------
if (Test-Path (Join-Path $RepoRoot '.git')) {
    git -C $RepoRoot config core.hooksPath .githooks 2>$null
}

# ---------- Kill existing process on port 8000 ----------
$portListeners = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($portListeners) {
    Info 'Port 8000 is in use — stopping existing process...'
    $portListeners | ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

# ---------- Start ----------
Ok 'Setup complete — starting Cadence server...'
& python -m cadence.server @ServerArgs
exit $LASTEXITCODE

$ErrorActionPreference = "SilentlyContinue"

# Resolve project root from script location so this works on any machine.
$ProjectRoot = $PSScriptRoot

function Write-Step($msg) { Write-Host "`n>> $msg" -ForegroundColor Cyan }
function Write-OK($msg)   { Write-Host "   [OK] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "   [!!] $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "   [XX] $msg" -ForegroundColor Red }

# ── Kill stale processes ──────────────────────────────────────────────────────
Write-Step "Cleaning up stale processes..."
foreach ($port in @(8000, 5173)) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        $conn | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
        Write-Warn "Killed old process on :$port"
    }
}
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# ── Clear log files safely (rename-then-delete avoids lock errors) ─────────────
Write-Step "Clearing log files..."
foreach ($log in @("backend.out.log", "backend.err.log", "frontend.out.log", "frontend.err.log")) {
    $path = Join-Path $ProjectRoot $log
    $tmp  = "$path.old"
    if (Test-Path $path) {
        Rename-Item $path $tmp -Force -ErrorAction SilentlyContinue
        Remove-Item $tmp  -Force -ErrorAction SilentlyContinue
    }
    $null = New-Item $path -ItemType File -Force -ErrorAction SilentlyContinue
}

# ── Docker Desktop ─────────────────────────────────────────────────────────────
Write-Step "Checking Docker Desktop..."
$dockerOk = $false
try {
    $info = docker info 2>&1
    if ($LASTEXITCODE -eq 0) { $dockerOk = $true }
} catch {}

if (-not $dockerOk) {
    Write-Warn "Docker Desktop not running - starting it..."
    $desktopExe = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $desktopExe) {
        Start-Process $desktopExe
    } else {
        Write-Fail "Docker Desktop not found at default path. Please start it manually."
        exit 1
    }
    Write-Host "   Waiting for Docker engine (up to 90s)..." -ForegroundColor Gray
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 1
        $info = docker info 2>&1
        if ($LASTEXITCODE -eq 0) { $dockerOk = $true; break }
        if (($i + 1) % 15 -eq 0) { Write-Host "   ...still waiting ($($i+1)s)" -ForegroundColor Gray }
    }
    if ($dockerOk) { Write-OK "Docker Desktop ready" }
    else { Write-Fail "Docker Desktop did not start in time. Please start it manually and re-run."; exit 1 }
} else {
    Write-OK "Docker Desktop already running"
}

# ── Qdrant (Docker) ────────────────────────────────────────────────────────────
Write-Step "Starting Qdrant via Docker..."
docker compose up -d qdrant 2>&1 | Out-Null

Write-Step "Waiting for Qdrant (port 6333)..."
$qdrantReady = $false
for ($i = 0; $i -lt 45; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:6333/readyz" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $qdrantReady = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}
if ($qdrantReady) {
    Write-OK "Qdrant ready at :6333"
} else {
    Write-Fail "Qdrant did not respond - check Docker Desktop"
    exit 1
}

# ── Guard: ensure .env points to Docker Qdrant, not a local path ───────────────
$envFile = Join-Path $ProjectRoot "backend\.env"
$envLines = Get-Content $envFile -ErrorAction SilentlyContinue
$needsPatch = $false
$envLines = $envLines | ForEach-Object {
    if ($_ -match "^AGENTBOOK_QDRANT_URL\s*=" -and $_ -notmatch "http://") {
        $needsPatch = $true
        "AGENTBOOK_QDRANT_URL=http://localhost:6333"
    } else {
        $_
    }
}
if ($needsPatch) {
    Set-Content $envFile $envLines -Encoding utf8
    Write-Warn ".env had a local Qdrant path - patched to http://localhost:6333"
}

# ── Backend ────────────────────────────────────────────────────────────────────
Write-Step "Starting Backend (Uvicorn on :8000)..."
$env:PYTHONIOENCODING = "utf-8"
# Allow EasyOCR + VietOCR + BGE-M3 (multiple torch stacks) to coexist without the
# Windows OpenMP duplicate-runtime crash (libiomp5 loaded more than once).
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

Start-Process -FilePath "python" `
    -ArgumentList "-m uvicorn src.main:app --port 8000" `
    -WorkingDirectory (Join-Path $ProjectRoot "backend") `
    -RedirectStandardOutput (Join-Path $ProjectRoot "backend.out.log") `
    -RedirectStandardError  (Join-Path $ProjectRoot "backend.err.log") `
    -WindowStyle Hidden

Write-Host "   Waiting for backend (up to 120s)..." -ForegroundColor Gray
$backendReady = $false
for ($i = 0; $i -lt 120; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $backendReady = $true; break }
    } catch {}
    $elapsed = $i + 1
    if ($elapsed % 20 -eq 0) {
        Write-Host "   ...still loading ($elapsed s)" -ForegroundColor Gray
        $lastErr = Get-Content (Join-Path $ProjectRoot "backend.err.log") -Tail 2 -ErrorAction SilentlyContinue
        if ($lastErr) { $lastErr | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray } }
    }
    Start-Sleep -Seconds 1
}

if ($backendReady) {
    Write-OK "Backend ready"
} else {
    Write-Fail "Backend did not respond in 120s"
    Write-Host ""
    Write-Host "   Last errors from backend.err.log:" -ForegroundColor Yellow
    Get-Content (Join-Path $ProjectRoot "backend.err.log") -Tail 20 -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Host "   $_" -ForegroundColor Red }
    exit 1
}

# ── Frontend ───────────────────────────────────────────────────────────────────
Write-Step "Starting Frontend (Vite on :5173)..."
Start-Process -FilePath "npm.cmd" `
    -ArgumentList "run dev" `
    -WorkingDirectory (Join-Path $ProjectRoot "frontend") `
    -RedirectStandardOutput (Join-Path $ProjectRoot "frontend.out.log") `
    -RedirectStandardError  (Join-Path $ProjectRoot "frontend.err.log") `
    -WindowStyle Hidden

$frontendReady = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:5173" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $frontendReady = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}
if ($frontendReady) {
    Write-OK "Frontend ready"
} else {
    Write-Warn "Frontend not responding - check frontend.err.log"
}

# ── Summary ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor White
if ($backendReady -and $frontendReady) {
    Write-Host "  All services UP!" -ForegroundColor Green
} elseif ($backendReady) {
    Write-Host "  Backend UP, frontend check failed" -ForegroundColor Yellow
} else {
    Write-Host "  Some services failed - check logs" -ForegroundColor Red
}
Write-Host "========================================"
Write-Host "  Backend  : http://localhost:8000"
Write-Host "  Docs     : http://localhost:8000/docs"
Write-Host "  Frontend : http://localhost:5173"
Write-Host "  Qdrant   : http://localhost:6333/dashboard"
Write-Host ""
Write-Host "  Logs: backend.err.log  /  frontend.err.log"
Write-Host "========================================"

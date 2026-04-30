$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "`n>> $msg" -ForegroundColor Cyan }
function Write-OK($msg)   { Write-Host "   [OK] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "   [!!] $msg" -ForegroundColor Yellow }

# --- Docker: Qdrant only (Redis not needed — Celery runs eager) ---
Write-Step "Starting Qdrant..."
$ErrorActionPreference = "SilentlyContinue"
docker compose up -d qdrant | Out-Null
$ErrorActionPreference = "Stop"

Write-Step "Waiting for Qdrant (port 6333)..."
$qdrantReady = $false
for ($i = 0; $i -lt 20; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:6333/readyz" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $qdrantReady = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}
if ($qdrantReady) { Write-OK "Qdrant ready" } else { Write-Warn "Qdrant did not respond in time — check Docker Desktop" }

# --- Backend ---
Write-Step "Starting Backend (Uvicorn on :8000)..."
$env:PYTHONIOENCODING = "utf-8"
Start-Process -FilePath "python" `
    -ArgumentList "-m uvicorn src.main:app --reload --port 8000" `
    -WorkingDirectory "d:\GenAI\DoAn01\backend" `
    -RedirectStandardOutput "d:\GenAI\DoAn01\backend.out.log" `
    -RedirectStandardError  "d:\GenAI\DoAn01\backend.err.log" `
    -WindowStyle Hidden

Start-Sleep -Seconds 4
$backendReady = $false
for ($i = 0; $i -lt 15; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $backendReady = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}
if ($backendReady) { Write-OK "Backend ready" } else { Write-Warn "Backend health check failed — check backend.err.log" }

# --- Frontend ---
Write-Step "Starting Frontend (Vite)..."
Start-Process -FilePath "npm.cmd" `
    -ArgumentList "run dev" `
    -WorkingDirectory "d:\GenAI\DoAn01\frontend" `
    -RedirectStandardOutput "d:\GenAI\DoAn01\frontend.out.log" `
    -RedirectStandardError  "d:\GenAI\DoAn01\frontend.err.log" `
    -WindowStyle Hidden

Start-Sleep -Seconds 3
Write-OK "Frontend launched"

# --- Summary ---
Write-Host "`n========================================" -ForegroundColor White
Write-Host "  All services started!" -ForegroundColor White
Write-Host "========================================" -ForegroundColor White
Write-Host "  Backend API  : http://localhost:8000"
Write-Host "  API Docs     : http://localhost:8000/docs"
Write-Host "  Qdrant UI    : http://localhost:6333/dashboard"
Write-Host "  Frontend     : http://localhost:5173"
Write-Host ""
Write-Host "  Logs:"
Write-Host "    backend.out.log / backend.err.log"
Write-Host "    frontend.out.log / frontend.err.log"
Write-Host "========================================`n" -ForegroundColor White

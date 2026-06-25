<#
.SYNOPSIS
    Quản lý AgentBook/Noelys chạy bằng Docker Compose.

.DESCRIPTION
    Script DUY NHẤT để vận hành dự án. Mọi service (frontend, api, worker,
    qdrant, redis) chạy trong Docker. Ollama chạy trên host (cho LLM local).

.PARAMETER Action
    up       (mặc định) Khởi động toàn bộ stack
    setup    Bootstrap máy mới: pull model Ollama + build + khởi động (1 lệnh)
    down     Tắt stack (giữ data)
    restart  Khởi động lại (mặc định api; truyền tên service để chỉ định)
    build    Build lại image rồi khởi động (sau khi sửa code)
    logs     Xem log realtime (mặc định api; truyền tên service để chỉ định)
    status   Xem trạng thái + endpoint
    stop     Dừng nhưng không xóa container

.PARAMETER Gpu
    Bật chế độ GPU (NVIDIA) — gộp thêm docker-compose.gpu.yml.
    Embedding/reranker/SigLIP chạy CUDA + FP16; Ollama tự dùng GPU trên host.

.EXAMPLE
    .\run.ps1                  # khởi động (CPU)
    .\run.ps1 setup -Gpu       # máy 4060 mới: pull model + build CUDA + chạy
    .\run.ps1 build -Gpu       # build lại image CUDA + khởi động
    .\run.ps1 logs worker      # xem log worker
    .\run.ps1 down             # tắt
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('up', 'setup', 'down', 'restart', 'build', 'logs', 'status', 'stop')]
    [string]$Action = 'up',

    [Parameter(Position = 1)]
    [string]$Service = '',

    [switch]$Gpu
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

# Gộp file override GPU khi -Gpu (giữ máy CPU không đổi gì).
$ComposeArgs = @('compose')
if ($Gpu) { $ComposeArgs += @('-f', 'docker-compose.yml', '-f', 'docker-compose.gpu.yml') }
function Invoke-Compose { docker @ComposeArgs @args }

$AppUrl     = 'http://localhost'
$ApiUrl     = 'http://localhost:8000'
$OllamaUrl  = 'http://localhost:11434'

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "[OK]  $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[!]   $msg" -ForegroundColor Yellow }

function Test-Ollama {
    try {
        $r = Invoke-RestMethod -Uri "$OllamaUrl/api/tags" -TimeoutSec 3
        $hasModel = $r.models | Where-Object { $_.name -like 'qwen2.5:7b*' }
        if ($hasModel) { Write-Ok "Ollama đang chạy (có qwen2.5:7b)" }
        else {
            Write-Warn "Ollama chạy nhưng THIẾU model qwen2.5:7b"
            Write-Host "      Pull bằng: ollama pull qwen2.5:7b" -ForegroundColor DarkGray
        }
    }
    catch {
        Write-Warn "Ollama CHƯA chạy ($OllamaUrl) — LLM local sẽ lỗi."
        Write-Host "      Mở app Ollama hoặc chạy: ollama serve" -ForegroundColor DarkGray
        Write-Host "      (Upload/parse/index vẫn chạy được nếu chưa cần hỏi đáp)" -ForegroundColor DarkGray
    }
}

function Install-OllamaModel {
    # Pull model nếu chưa có — chạy lúc setup máy mới.
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Write-Warn 'Chưa cài Ollama trên host. Tải tại https://ollama.com rồi chạy lại.'
        return
    }
    try { $r = Invoke-RestMethod -Uri "$OllamaUrl/api/tags" -TimeoutSec 3 } catch { $r = $null }
    $hasModel = $r.models | Where-Object { $_.name -like 'qwen2.5:7b*' }
    if ($hasModel) { Write-Ok 'Model qwen2.5:7b đã có.'; return }
    Write-Step 'Pull model qwen2.5:7b (~4.7GB, chỉ lần đầu)'
    ollama pull qwen2.5:7b
    Write-Ok 'Đã pull qwen2.5:7b.'
}

function Show-Status {
    Write-Step 'Trạng thái services'
    Invoke-Compose ps --format "table {{.Service}}`t{{.Status}}`t{{.Ports}}"
    Write-Host ''
    Write-Step 'Kiểm tra endpoint'
    foreach ($e in @(
        @{ Name = 'Frontend'; Url = $AppUrl },
        @{ Name = 'API /health'; Url = "$ApiUrl/health" },
        @{ Name = 'API /docs'; Url = "$ApiUrl/docs" }
    )) {
        try {
            $code = (Invoke-WebRequest -Uri $e.Url -TimeoutSec 8 -UseBasicParsing).StatusCode
            Write-Ok ("{0,-14} HTTP {1}" -f $e.Name, $code)
        }
        catch { Write-Warn ("{0,-14} không phản hồi" -f $e.Name) }
    }
}

switch ($Action) {
    'up' {
        Write-Step 'Kiểm tra Ollama (host)'
        Test-Ollama
        Write-Step ("Khởi động Docker stack" + $(if ($Gpu) { ' (GPU)' } else { '' }))
        Invoke-Compose up -d
        Write-Host ''
        Write-Ok "Đã khởi động. Đợi ~30-60s cho services healthy."
        Write-Host ''
        Write-Host "  Frontend : $AppUrl"          -ForegroundColor White
        Write-Host "  API docs : $ApiUrl/docs"      -ForegroundColor White
        Write-Host "  Qdrant   : http://localhost:6333/dashboard" -ForegroundColor White
        Write-Host ''
        Write-Host "  Xem trạng thái: .\run.ps1 status" -ForegroundColor DarkGray
        Write-Host "  Xem log       : .\run.ps1 logs"   -ForegroundColor DarkGray
    }
    'setup' {
        Write-Step ('Bootstrap máy mới' + $(if ($Gpu) { ' (GPU)' } else { ' (CPU)' }))
        Install-OllamaModel
        Write-Step 'Build image + khởi động'
        Invoke-Compose up -d --build
        Write-Host ''
        Write-Ok 'Setup xong. Lần request đầu sẽ chậm (api tải BGE-M3 + model HF về ./data).'
        Write-Host '  Mẹo chép nhanh model HF: copy thư mục ./data/.hf_cache từ máy cũ sang để khỏi tải lại.' -ForegroundColor DarkGray
        Write-Host "  Xem trạng thái: .\run.ps1 status$(if ($Gpu) { ' -Gpu' })" -ForegroundColor DarkGray
    }
    'down' {
        Write-Step 'Tắt stack (giữ data)'
        Invoke-Compose down
        Write-Ok 'Đã tắt. Data được giữ (volume ./data + redis-data).'
    }
    'stop' {
        Write-Step 'Dừng container (không xóa)'
        Invoke-Compose stop
        Write-Ok 'Đã dừng. Chạy lại: .\run.ps1 up'
    }
    'restart' {
        $svc = if ($Service) { $Service } else { 'api' }
        Write-Step "Khởi động lại: $svc"
        Invoke-Compose restart $svc
        Write-Ok "Đã restart $svc"
    }
    'build' {
        Write-Step ('Build lại image + khởi động' + $(if ($Gpu) { ' (GPU)' } else { '' }))
        Test-Ollama
        if ($Service) { Invoke-Compose up -d --build $Service }
        else { Invoke-Compose up -d --build }
        Write-Ok 'Build + khởi động xong.'
    }
    'logs' {
        $svc = if ($Service) { $Service } else { 'api' }
        Write-Step "Log realtime: $svc  (Ctrl+C để thoát)"
        Invoke-Compose logs $svc -f --tail=50
    }
    'status' {
        Show-Status
        Write-Host ''
        Test-Ollama
    }
}

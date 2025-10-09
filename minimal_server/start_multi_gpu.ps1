#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Start multiple gRPC servers across multiple GPUs with flexible configuration

.DESCRIPTION
    Launches X processes per GPU across Y GPUs for maximum throughput.
    Works with both consumer GPUs (1-4x gain) and professional GPUs (3-7x gain).
    Automatically detects GPU count and adjusts configuration.

.PARAMETER NumGPUs
    Number of GPUs to use (default: auto-detect, max: 8)

.PARAMETER ProcessesPerGPU
    Number of server processes per GPU (default: 6)
    Recommended: 4-6 for consumer GPUs, 6-8 for professional GPUs

.PARAMETER BasePort
    Starting port number (default: 50051)

.PARAMETER EnableMPS
    Enable NVIDIA MPS (Multi-Process Service) for better concurrency
    Only works on Linux/WSL2. Ignored on Windows.

.EXAMPLE
    .\start_multi_gpu.ps1
    # Auto-detect GPUs, start 6 processes per GPU

.EXAMPLE
    .\start_multi_gpu.ps1 -NumGPUs 4 -ProcessesPerGPU 8
    # Use 4 GPUs, 8 processes each = 32 total servers

.EXAMPLE
    .\start_multi_gpu.ps1 -NumGPUs 1 -ProcessesPerGPU 4
    # Single GPU with 4 processes (good for testing)

.EXAMPLE
    .\start_multi_gpu.ps1 -NumGPUs 8 -ProcessesPerGPU 6 -EnableMPS
    # 8 GPUs, 6 processes each = 48 servers with MPS enabled
#>

param(
    [int]$NumGPUs = -1,  # -1 means auto-detect
    [int]$ProcessesPerGPU = 6,
    [int]$BasePort = 50051,
    [switch]$EnableMPS = $false
)

# Colors for output
$ColorInfo = "Cyan"
$ColorSuccess = "Green"
$ColorWarning = "Yellow"
$ColorError = "Red"

Write-Host ""
Write-Host "================================================================================" -ForegroundColor $ColorInfo
Write-Host "MULTI-GPU GRPC SERVER LAUNCHER" -ForegroundColor $ColorInfo
Write-Host "================================================================================" -ForegroundColor $ColorInfo
Write-Host ""

# Detect available GPUs
Write-Host "🔍 Detecting GPUs..." -ForegroundColor $ColorInfo
try {
    $gpuOutput = nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Error: nvidia-smi failed. Is NVIDIA driver installed?" -ForegroundColor $ColorError
        exit 1
    }
    
    $availableGPUs = @($gpuOutput | ForEach-Object { $_ -split ',' | Select-Object -First 1 })
    $totalGPUs = $availableGPUs.Count
    
    Write-Host "   Found $totalGPUs GPU(s):" -ForegroundColor $ColorSuccess
    
    $gpuInfo = nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    $gpuInfo | ForEach-Object {
        Write-Host "   $_" -ForegroundColor $ColorSuccess
    }
    
    # Use specified or detected GPU count
    if ($NumGPUs -eq -1) {
        $NumGPUs = $totalGPUs
        Write-Host ""
        Write-Host "   Auto-detected: Using all $NumGPUs GPU(s)" -ForegroundColor $ColorInfo
    } elseif ($NumGPUs -gt $totalGPUs) {
        Write-Host ""
        Write-Host "   ⚠️  Requested $NumGPUs GPUs but only $totalGPUs available" -ForegroundColor $ColorWarning
        Write-Host "   Using $totalGPUs GPU(s)" -ForegroundColor $ColorWarning
        $NumGPUs = $totalGPUs
    }
    
} catch {
    Write-Host "❌ Error detecting GPUs: $_" -ForegroundColor $ColorError
    exit 1
}

# Detect GPU type (consumer vs professional)
Write-Host ""
Write-Host "🔍 Analyzing GPU architecture..." -ForegroundColor $ColorInfo

$gpuNames = nvidia-smi --query-gpu=name --format=csv,noheader
$isProfessional = $false

foreach ($name in $gpuNames) {
    if ($name -match "RTX \d000|A\d+|H\d+|Tesla|Quadro") {
        $isProfessional = $true
        break
    }
}

if ($isProfessional) {
    Write-Host "   ✅ Professional GPU detected (RTX 6000/A100/H100/Tesla/Quadro)" -ForegroundColor $ColorSuccess
    Write-Host "   Expected speedup: 3-7x with multi-process" -ForegroundColor $ColorSuccess
} else {
    Write-Host "   ℹ️  Consumer GPU detected (RTX 2060-4090)" -ForegroundColor $ColorInfo
    Write-Host "   Expected speedup: 1.5-2.5x with multi-process (time-slicing)" -ForegroundColor $ColorInfo
    Write-Host "   Tip: Professional GPUs (RTX 6000 Ada) provide better multi-process support" -ForegroundColor $ColorInfo
}

# MPS check (Linux only)
if ($EnableMPS) {
    if ($IsLinux) {
        Write-Host ""
        Write-Host "🔧 Enabling NVIDIA MPS (Multi-Process Service)..." -ForegroundColor $ColorInfo
        try {
            $env:CUDA_VISIBLE_DEVICES = ""
            & nvidia-cuda-mps-control -d 2>&1 | Out-Null
            Write-Host "   ✅ MPS enabled" -ForegroundColor $ColorSuccess
        } catch {
            Write-Host "   ⚠️  Failed to enable MPS: $_" -ForegroundColor $ColorWarning
            Write-Host "   Continuing without MPS..." -ForegroundColor $ColorWarning
        }
    } else {
        Write-Host ""
        Write-Host "   ℹ️  MPS is only supported on Linux/WSL2 (ignored on Windows)" -ForegroundColor $ColorInfo
    }
}

# Calculate total configuration
$totalServers = $NumGPUs * $ProcessesPerGPU
$endPort = $BasePort + $totalServers - 1

Write-Host ""
Write-Host "================================================================================" -ForegroundColor $ColorInfo
Write-Host "CONFIGURATION" -ForegroundColor $ColorInfo
Write-Host "================================================================================" -ForegroundColor $ColorInfo
Write-Host "   GPUs:                $NumGPUs" -ForegroundColor $ColorInfo
Write-Host "   Processes per GPU:   $ProcessesPerGPU" -ForegroundColor $ColorInfo
Write-Host "   Total servers:       $totalServers" -ForegroundColor $ColorSuccess
Write-Host "   Port range:          $BasePort - $endPort" -ForegroundColor $ColorInfo
Write-Host "   Delay between starts: 8 seconds" -ForegroundColor $ColorInfo
Write-Host ""

# Estimate performance
$baselineFPS = 58
if ($isProfessional) {
    $minFPS = [int]($baselineFPS * $ProcessesPerGPU * 0.5 * $NumGPUs)
    $maxFPS = [int]($baselineFPS * $ProcessesPerGPU * 0.7 * $NumGPUs)
} else {
    $minFPS = [int]($baselineFPS * $ProcessesPerGPU * 0.25 * $NumGPUs)
    $maxFPS = [int]($baselineFPS * $ProcessesPerGPU * 0.35 * $NumGPUs)
}

Write-Host "📊 Expected Performance:" -ForegroundColor $ColorInfo
Write-Host "   Single GPU baseline:  $baselineFPS FPS" -ForegroundColor $ColorInfo
Write-Host "   Estimated total:      $minFPS - $maxFPS FPS" -ForegroundColor $ColorSuccess
Write-Host ""

# Confirm before proceeding
Write-Host "Press Enter to start $totalServers servers, or Ctrl+C to cancel..." -ForegroundColor $ColorWarning
$null = Read-Host

Write-Host ""
Write-Host "================================================================================" -ForegroundColor $ColorInfo
Write-Host "STARTING SERVERS" -ForegroundColor $ColorInfo
Write-Host "================================================================================" -ForegroundColor $ColorInfo
Write-Host ""

# Track PIDs
$allPIDs = @()
$port = $BasePort

# Launch servers across all GPUs
for ($gpu = 0; $gpu -lt $NumGPUs; $gpu++) {
    Write-Host "🖥️  GPU $gpu - Starting $ProcessesPerGPU processes..." -ForegroundColor $ColorInfo
    Write-Host ""
    
    for ($i = 0; $i -lt $ProcessesPerGPU; $i++) {
        $serverNum = ($gpu * $ProcessesPerGPU) + $i + 1
        
        Write-Host "   [$serverNum/$totalServers] GPU $gpu, Port $port - Starting..." -ForegroundColor $ColorInfo
        
        # Start process with CUDA_VISIBLE_DEVICES set to specific GPU
        $process = Start-Process powershell -ArgumentList @(
            "-NoExit",
            "-Command",
            "& { `$env:CUDA_VISIBLE_DEVICES=$gpu; `$host.ui.RawUI.WindowTitle='gRPC Server - GPU $gpu - Port $port'; python optimized_grpc_server.py --port $port }"
        ) -WindowStyle Minimized -PassThru
        
        $allPIDs += $process.Id
        
        Write-Host "   [$serverNum/$totalServers] GPU $gpu, Port $port - PID $($process.Id)" -ForegroundColor $ColorSuccess
        
        $port++
        
        # Delay between starts (allow model loading)
        if ($serverNum -lt $totalServers) {
            Start-Sleep -Seconds 8
        }
    }
    
    Write-Host ""
}

# Save PIDs to file
$pidFile = "grpc_processes.txt"
$allPIDs | Out-File -FilePath $pidFile -Encoding UTF8

Write-Host "================================================================================" -ForegroundColor $ColorSuccess
Write-Host "✅ ALL SERVERS STARTED" -ForegroundColor $ColorSuccess
Write-Host "================================================================================" -ForegroundColor $ColorSuccess
Write-Host ""
Write-Host "📊 Summary:" -ForegroundColor $ColorInfo
Write-Host "   Total servers:    $totalServers" -ForegroundColor $ColorSuccess
Write-Host "   GPUs used:        $NumGPUs" -ForegroundColor $ColorSuccess
Write-Host "   Ports:            $BasePort - $endPort" -ForegroundColor $ColorSuccess
Write-Host "   PIDs saved to:    $pidFile" -ForegroundColor $ColorInfo
Write-Host ""
Write-Host "⏳ Wait ~30-45 seconds for all models to load..." -ForegroundColor $ColorWarning
Write-Host ""
Write-Host "================================================================================" -ForegroundColor $ColorInfo
Write-Host "NEXT STEPS" -ForegroundColor $ColorInfo
Write-Host "================================================================================" -ForegroundColor $ColorInfo
Write-Host ""
Write-Host "1. Start the Go proxy:" -ForegroundColor $ColorInfo
Write-Host "   cd ..\grpc-websocket-proxy" -ForegroundColor $ColorSuccess
Write-Host "   .\proxy.exe --start-port $BasePort --num-servers $totalServers" -ForegroundColor $ColorSuccess
Write-Host ""
Write-Host "2. Open browser client:" -ForegroundColor $ColorInfo
Write-Host "   http://localhost:8086/" -ForegroundColor $ColorSuccess
Write-Host ""
Write-Host "3. Test performance:" -ForegroundColor $ColorInfo
Write-Host "   python test_multi_process.py --ports $BasePort-$endPort" -ForegroundColor $ColorSuccess
Write-Host ""
Write-Host "4. Monitor GPUs:" -ForegroundColor $ColorInfo
Write-Host "   nvidia-smi dmon -s um -c 30" -ForegroundColor $ColorSuccess
Write-Host ""
Write-Host "5. Check health:" -ForegroundColor $ColorInfo
Write-Host "   curl http://localhost:8086/health" -ForegroundColor $ColorSuccess
Write-Host ""
Write-Host "================================================================================" -ForegroundColor $ColorInfo
Write-Host "TO STOP ALL SERVERS" -ForegroundColor $ColorInfo
Write-Host "================================================================================" -ForegroundColor $ColorInfo
Write-Host ""
Write-Host "Stop-Process -Id (Get-Content $pidFile)" -ForegroundColor $ColorWarning
Write-Host ""

# Health monitoring loop (optional)
Write-Host "Press Ctrl+C to exit monitoring..." -ForegroundColor $ColorInfo
Write-Host ""

$startTime = Get-Date

try {
    while ($true) {
        Start-Sleep -Seconds 10
        
        $elapsed = ((Get-Date) - $startTime).TotalSeconds
        $running = 0
        
        foreach ($processId in $allPIDs) {
            if (Get-Process -Id $processId -ErrorAction SilentlyContinue) {
                $running++
            }
        }
        
        if ($running -eq $totalServers) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] ✅ All $totalServers servers running ($([int]$elapsed)s elapsed)" -ForegroundColor $ColorSuccess
        } else {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] ⚠️  $running/$totalServers servers running ($(($totalServers - $running)) stopped)" -ForegroundColor $ColorWarning
        }
    }
} catch {
    Write-Host ""
    Write-Host "Monitoring stopped." -ForegroundColor $ColorInfo
}

# Multi-Process gRPC Server Launcher for RTX 6000 Ada
# Optimized for professional GPU with better multi-process support

param(
    [int]$NumProcesses = 4,
    [int]$StartPort = 50051,
    [string]$PythonPath = "D:\Projects\webcodecstest\.venv312\Scripts\python.exe",
    [string]$ServerScript = "optimized_grpc_server.py"
)

Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "🚀 MULTI-PROCESS GRPC SERVER LAUNCHER" -ForegroundColor Cyan
Write-Host "   Optimized for RTX 6000 Ada Blackwell" -ForegroundColor Cyan
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python exists
if (-not (Test-Path $PythonPath)) {
    Write-Host "❌ Python not found at: $PythonPath" -ForegroundColor Red
    Write-Host "   Update the `$PythonPath variable or run from .venv" -ForegroundColor Yellow
    exit 1
}

# Check if server script exists
if (-not (Test-Path $ServerScript)) {
    Write-Host "❌ Server script not found: $ServerScript" -ForegroundColor Red
    Write-Host "   Make sure you're in the minimal_server directory" -ForegroundColor Yellow
    exit 1
}

Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  Processes: $NumProcesses" -ForegroundColor White
Write-Host "  Port range: $StartPort-$($StartPort + $NumProcesses - 1)" -ForegroundColor White
Write-Host "  Python: $PythonPath" -ForegroundColor White
Write-Host ""

# Array to store process objects
$processes = @()

# Start each server process
for ($i = 0; $i -lt $NumProcesses; $i++) {
    $port = $StartPort + $i
    
    Write-Host "Starting server $($i + 1)/$NumProcesses on port $port..." -ForegroundColor Yellow
    
    $proc = Start-Process -FilePath $PythonPath `
                          -ArgumentList $ServerScript, "--port", $port `
                          -WorkingDirectory $PWD `
                          -PassThru `
                          -WindowStyle Normal
    
    $processes += $proc
    
    Write-Host "  ✅ PID: $($proc.Id)" -ForegroundColor Green
    
    # Wait between starts to allow model loading
    if ($i -lt $NumProcesses - 1) {
        Write-Host "  Waiting 8 seconds for model initialization..." -ForegroundColor Gray
        Start-Sleep -Seconds 8
    }
}

Write-Host ""
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "✅ ALL SERVERS STARTED!" -ForegroundColor Green
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Server ports:" -ForegroundColor White
for ($i = 0; $i -lt $NumProcesses; $i++) {
    $port = $StartPort + $i
    Write-Host "  Server $($i + 1): localhost:$port (PID: $($processes[$i].Id))" -ForegroundColor Cyan
}
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Monitor GPU usage: nvidia-smi dmon -s u -d 1" -ForegroundColor White
Write-Host "  2. Test servers: python test_multi_process.py" -ForegroundColor White
Write-Host "  3. Update Go proxy to use all ports" -ForegroundColor White
Write-Host ""
Write-Host "To stop all servers:" -ForegroundColor Yellow
Write-Host "  Get-Process python | Where-Object {`$_.MainWindowTitle -like '*grpc*'} | Stop-Process" -ForegroundColor White
Write-Host ""
Write-Host "Process IDs saved to: multi_grpc_pids.txt" -ForegroundColor Gray
$processes.Id | Out-File -FilePath "multi_grpc_pids.txt" -Encoding UTF8
Write-Host ""
Write-Host "Press Ctrl+C to exit (servers will continue running)" -ForegroundColor DarkGray

# Keep script running to show status
Write-Host ""
Write-Host "Monitoring server health (press Ctrl+C to stop monitoring)..." -ForegroundColor Cyan
Write-Host ""

try {
    while ($true) {
        Start-Sleep -Seconds 10
        
        $alive = 0
        $dead = 0
        
        foreach ($proc in $processes) {
            if (-not $proc.HasExited) {
                $alive++
            } else {
                $dead++
            }
        }
        
        $timestamp = Get-Date -Format "HH:mm:ss"
        if ($dead -eq 0) {
            Write-Host "[$timestamp] Status: $alive/$NumProcesses servers running ✅" -ForegroundColor Green
        } else {
            Write-Host "[$timestamp] Status: $alive/$NumProcesses servers running, $dead died ❌" -ForegroundColor Red
        }
    }
} catch {
    Write-Host ""
    Write-Host "Monitoring stopped. Servers are still running in background." -ForegroundColor Yellow
}

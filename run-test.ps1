# Test the separated architecture
# This script starts both servers and runs the test client

Write-Host "🚀 Starting Separated Architecture Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Kill any existing servers
Write-Host "🧹 Cleaning up existing servers..." -ForegroundColor Yellow
Get-Process | Where-Object { $_.ProcessName -like "*inference-server*" -or $_.ProcessName -like "*compositing-server*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# Start inference server in new window
Write-Host "🔧 Starting Inference Server (GPU)..." -ForegroundColor Green
$inferenceServer = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Projects\webcodecstest\go-inference-server; .\inference-server.exe" -PassThru -WindowStyle Normal
Start-Sleep -Seconds 3

# Start compositing server in new window
Write-Host "🎨 Starting Compositing Server (CPU)..." -ForegroundColor Green
$compositingServer = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Projects\webcodecstest\go-compositing-server; .\compositing-server.exe" -PassThru -WindowStyle Normal
Start-Sleep -Seconds 3

# Run test client
Write-Host ""
Write-Host "🧪 Running Test Client..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

cd D:\Projects\webcodecstest\go-compositing-server
.\test-client.exe

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test complete!" -ForegroundColor Green
Write-Host ""
Write-Host "The servers are still running in separate windows." -ForegroundColor Yellow
Write-Host "Close them manually when done, or run:" -ForegroundColor Yellow
Write-Host "Get-Process *server* | Stop-Process" -ForegroundColor Yellow
Write-Host ""

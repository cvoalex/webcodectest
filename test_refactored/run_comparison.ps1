# Comparison Test Runner for Original vs Refactored Server
# This script runs both tests and compares the results

$ErrorActionPreference = "Stop"

Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "🔬 MONOLITHIC SERVER REFACTORING - PERFORMANCE COMPARISON" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 69) -ForegroundColor Cyan

# Check prerequisites
Write-Host "`n📋 Checking prerequisites..." -ForegroundColor Yellow

# Check if Go is installed
if (-not (Get-Command go -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Go is not installed or not in PATH" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Go found: $(go version)" -ForegroundColor Green

# Check if Python is installed (needed for loading numpy files)
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Python found: $(python --version)" -ForegroundColor Green

# Check if numpy is installed
$numpyCheck = python -c "import numpy; print(numpy.__version__)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ NumPy is not installed" -ForegroundColor Red
    exit 1
}
Write-Host "✅ NumPy found: version $numpyCheck" -ForegroundColor Green

# Check if audio file exists
$audioFile = "..\go-monolithic-server\testing\aud.wav"
if (-not (Test-Path $audioFile)) {
    Write-Host "❌ Audio file not found: $audioFile" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Audio file found" -ForegroundColor Green

# Check if visual frames file exists
$visualFile = "..\go-monolithic-server\testing\visual_frames_6.npy"
if (-not (Test-Path $visualFile)) {
    Write-Host "❌ Visual frames file not found: $visualFile" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Visual frames file found" -ForegroundColor Green

Write-Host "`n📝 Test Configuration:" -ForegroundColor Yellow
Write-Host "   • Batch Size: 8 frames" -ForegroundColor White
Write-Host "   • Warmup Runs: 3 iterations" -ForegroundColor White
Write-Host "   • Timed Runs: 3 iterations (averaged)" -ForegroundColor White
Write-Host "   • Model: sanders" -ForegroundColor White
Write-Host "   • Audio: Real audio from aud.wav" -ForegroundColor White
Write-Host "   • Visual: Real frames from visual_frames_6.npy" -ForegroundColor White

# Prompt user for which test to run
Write-Host "`n❓ Which test would you like to run?" -ForegroundColor Yellow
Write-Host "   1. Test ORIGINAL server only (localhost:50053)" -ForegroundColor White
Write-Host "   2. Test REFACTORED server only (localhost:50054)" -ForegroundColor White
Write-Host "   3. Test BOTH servers and compare" -ForegroundColor White
$choice = Read-Host "Enter choice (1-3)"

switch ($choice) {
    "1" {
        Write-Host "`n🚀 Running ORIGINAL server test..." -ForegroundColor Cyan
        Write-Host "⚠️  Make sure the ORIGINAL server is running on localhost:50053" -ForegroundColor Yellow
        Write-Host "   (cd ..\go-monolithic-server && go run cmd/server/main.go)" -ForegroundColor Gray
        Read-Host "Press Enter when ready to start test"
        
        go run test_original.go
        
        Write-Host "`n✅ Test complete! Check output_original/ for results" -ForegroundColor Green
    }
    
    "2" {
        Write-Host "`n🚀 Running REFACTORED server test..." -ForegroundColor Cyan
        Write-Host "⚠️  Make sure the REFACTORED server is running on localhost:50054" -ForegroundColor Yellow
        Write-Host "   (cd ..\go-monolithic-server-refactored && go run cmd/server/main.go)" -ForegroundColor Gray
        Read-Host "Press Enter when ready to start test"
        
        go run test_refactored.go
        
        Write-Host "`n✅ Test complete! Check output_refactored/ for results" -ForegroundColor Green
    }
    
    "3" {
        Write-Host "`n🚀 Running BOTH server tests for comparison..." -ForegroundColor Cyan
        
        # Test original server
        Write-Host "`n📍 STEP 1/2: Testing ORIGINAL server" -ForegroundColor Yellow
        Write-Host "⚠️  Make sure the ORIGINAL server is running on localhost:50053" -ForegroundColor Yellow
        Write-Host "   (cd ..\go-monolithic-server && go run cmd/server/main.go)" -ForegroundColor Gray
        Read-Host "Press Enter when ready to start test"
        
        Write-Host "`n🔵 Running original server test..." -ForegroundColor Cyan
        $originalOutput = go run test_original.go 2>&1
        Write-Host $originalOutput
        
        # Extract metrics from original
        $originalFPS = if ($originalOutput -match "Average FPS: ([\d.]+)") { [decimal]$matches[1] } else { 0 }
        $originalAudio = if ($originalOutput -match "Average Audio Processing: ([\d.]+) ms") { [decimal]$matches[1] } else { 0 }
        $originalInfer = if ($originalOutput -match "Average Inference: ([\d.]+) ms") { [decimal]$matches[1] } else { 0 }
        $originalComposite = if ($originalOutput -match "Average Compositing: ([\d.]+) ms") { [decimal]$matches[1] } else { 0 }
        $originalTotal = $originalAudio + $originalInfer + $originalComposite
        
        Write-Host "`n✅ Original server test complete!" -ForegroundColor Green
        Write-Host "   Please STOP the original server and START the refactored server" -ForegroundColor Yellow
        
        # Test refactored server
        Write-Host "`n📍 STEP 2/2: Testing REFACTORED server" -ForegroundColor Yellow
        Write-Host "⚠️  Make sure the REFACTORED server is running on localhost:50054" -ForegroundColor Yellow
        Write-Host "   (cd ..\go-monolithic-server-refactored && go run cmd/server/main.go)" -ForegroundColor Gray
        Read-Host "Press Enter when ready to start test"
        
        Write-Host "`n🟢 Running refactored server test..." -ForegroundColor Cyan
        $refactoredOutput = go run test_refactored.go 2>&1
        Write-Host $refactoredOutput
        
        # Extract metrics from refactored
        $refactoredFPS = if ($refactoredOutput -match "Average FPS: ([\d.]+)") { [decimal]$matches[1] } else { 0 }
        $refactoredAudio = if ($refactoredOutput -match "Average Audio Processing: ([\d.]+) ms") { [decimal]$matches[1] } else { 0 }
        $refactoredInfer = if ($refactoredOutput -match "Average Inference: ([\d.]+) ms") { [decimal]$matches[1] } else { 0 }
        $refactoredComposite = if ($refactoredOutput -match "Average Compositing: ([\d.]+) ms") { [decimal]$matches[1] } else { 0 }
        $refactoredTotal = $refactoredAudio + $refactoredInfer + $refactoredComposite
        
        Write-Host "`n✅ Refactored server test complete!" -ForegroundColor Green
        
        # Compare results
        Write-Host "`n" -NoNewline
        Write-Host ("=" * 70) -ForegroundColor Cyan
        Write-Host "📊 PERFORMANCE COMPARISON RESULTS" -ForegroundColor Cyan
        Write-Host ("=" * 70) -ForegroundColor Cyan
        
        Write-Host "`n┌─────────────────────────┬──────────────┬──────────────┬──────────────┐" -ForegroundColor White
        Write-Host "│ Metric                  │ Original     │ Refactored   │ Difference   │" -ForegroundColor White
        Write-Host "├─────────────────────────┼──────────────┼──────────────┼──────────────┤" -ForegroundColor White
        
        # FPS comparison
        $fpsDiff = $refactoredFPS - $originalFPS
        $fpsPercent = if ($originalFPS -gt 0) { ($fpsDiff / $originalFPS) * 100 } else { 0 }
        $fpsColor = if ($fpsDiff -ge 0) { "Green" } else { "Red" }
        $fpsSign = if ($fpsDiff -ge 0) { "+" } else { "" }
        Write-Host "│ Throughput (FPS)        │ " -NoNewline -ForegroundColor White
        Write-Host ("{0,12:N2}" -f $originalFPS) -NoNewline -ForegroundColor White
        Write-Host " │ " -NoNewline -ForegroundColor White
        Write-Host ("{0,12:N2}" -f $refactoredFPS) -NoNewline -ForegroundColor White
        Write-Host " │ " -NoNewline -ForegroundColor White
        Write-Host ("{0}{1,8:N2} " -f $fpsSign, $fpsDiff) -NoNewline -ForegroundColor $fpsColor
        Write-Host ("({0}{1,5:N1}%)" -f $fpsSign, $fpsPercent) -NoNewline -ForegroundColor $fpsColor
        Write-Host " │" -ForegroundColor White
        
        # Audio processing comparison
        $audioDiff = $refactoredAudio - $originalAudio
        $audioColor = if ($audioDiff -le 0) { "Green" } else { "Red" }
        $audioSign = if ($audioDiff -ge 0) { "+" } else { "" }
        Write-Host "│ Audio Processing (ms)   │ " -NoNewline -ForegroundColor White
        Write-Host ("{0,12:N2}" -f $originalAudio) -NoNewline -ForegroundColor White
        Write-Host " │ " -NoNewline -ForegroundColor White
        Write-Host ("{0,12:N2}" -f $refactoredAudio) -NoNewline -ForegroundColor White
        Write-Host " │ " -NoNewline -ForegroundColor White
        Write-Host ("{0}{1,12:N2}" -f $audioSign, $audioDiff) -NoNewline -ForegroundColor $audioColor
        Write-Host " │" -ForegroundColor White
        
        # Inference comparison
        $inferDiff = $refactoredInfer - $originalInfer
        $inferColor = if ($inferDiff -le 0) { "Green" } else { "Red" }
        $inferSign = if ($inferDiff -ge 0) { "+" } else { "" }
        Write-Host "│ Inference (ms)          │ " -NoNewline -ForegroundColor White
        Write-Host ("{0,12:N2}" -f $originalInfer) -NoNewline -ForegroundColor White
        Write-Host " │ " -NoNewline -ForegroundColor White
        Write-Host ("{0,12:N2}" -f $refactoredInfer) -NoNewline -ForegroundColor White
        Write-Host " │ " -NoNewline -ForegroundColor White
        Write-Host ("{0}{1,12:N2}" -f $inferSign, $inferDiff) -NoNewline -ForegroundColor $inferColor
        Write-Host " │" -ForegroundColor White
        
        # Compositing comparison
        $compositeDiff = $refactoredComposite - $originalComposite
        $compositeColor = if ($compositeDiff -le 0) { "Green" } else { "Red" }
        $compositeSign = if ($compositeDiff -ge 0) { "+" } else { "" }
        Write-Host "│ Compositing (ms)        │ " -NoNewline -ForegroundColor White
        Write-Host ("{0,12:N2}" -f $originalComposite) -NoNewline -ForegroundColor White
        Write-Host " │ " -NoNewline -ForegroundColor White
        Write-Host ("{0,12:N2}" -f $refactoredComposite) -NoNewline -ForegroundColor White
        Write-Host " │ " -NoNewline -ForegroundColor White
        Write-Host ("{0}{1,12:N2}" -f $compositeSign, $compositeDiff) -NoNewline -ForegroundColor $compositeColor
        Write-Host " │" -ForegroundColor White
        
        # Total time comparison
        $totalDiff = $refactoredTotal - $originalTotal
        $totalColor = if ($totalDiff -le 0) { "Green" } else { "Red" }
        $totalSign = if ($totalDiff -ge 0) { "+" } else { "" }
        Write-Host "│ Total Time (ms)         │ " -NoNewline -ForegroundColor White
        Write-Host ("{0,12:N2}" -f $originalTotal) -NoNewline -ForegroundColor White
        Write-Host " │ " -NoNewline -ForegroundColor White
        Write-Host ("{0,12:N2}" -f $refactoredTotal) -NoNewline -ForegroundColor White
        Write-Host " │ " -NoNewline -ForegroundColor White
        Write-Host ("{0}{1,12:N2}" -f $totalSign, $totalDiff) -NoNewline -ForegroundColor $totalColor
        Write-Host " │" -ForegroundColor White
        
        Write-Host "└─────────────────────────┴──────────────┴──────────────┴──────────────┘" -ForegroundColor White
        
        # Verdict
        Write-Host "`n🎯 VERDICT:" -ForegroundColor Yellow
        $threshold = 0.5 # 0.5 FPS tolerance
        if ([Math]::Abs($fpsDiff) -le $threshold) {
            Write-Host "✅ PERFORMANCE IDENTICAL - Refactoring successful!" -ForegroundColor Green
            Write-Host "   The refactored server performs within $threshold FPS of the original." -ForegroundColor Green
        } elseif ($fpsDiff -gt $threshold) {
            Write-Host "🚀 PERFORMANCE IMPROVED - Refactoring beneficial!" -ForegroundColor Green
            Write-Host "   The refactored server is ${fpsDiff:N2} FPS faster (+${fpsPercent:N1}%)." -ForegroundColor Green
        } else {
            Write-Host "⚠️  PERFORMANCE REGRESSION DETECTED" -ForegroundColor Red
            Write-Host "   The refactored server is ${fpsDiff:N2} FPS slower (${fpsPercent:N1}%)." -ForegroundColor Red
            Write-Host "   Investigation recommended." -ForegroundColor Yellow
        }
        
        # Image comparison
        Write-Host "`n🖼️  OUTPUT COMPARISON:" -ForegroundColor Yellow
        $originalFrame = "output_original\frame_0_original.jpg"
        $refactoredFrame = "output_refactored\frame_0_refactored.jpg"
        
        if ((Test-Path $originalFrame) -and (Test-Path $refactoredFrame)) {
            $originalSize = (Get-Item $originalFrame).Length
            $refactoredSize = (Get-Item $refactoredFrame).Length
            $sizeDiff = $refactoredSize - $originalSize
            $sizePercent = if ($originalSize -gt 0) { ($sizeDiff / $originalSize) * 100 } else { 0 }
            
            Write-Host "   Original Frame:    $originalSize bytes" -ForegroundColor White
            Write-Host "   Refactored Frame:  $refactoredSize bytes (diff: $sizeDiff bytes, ${sizePercent:N2}%)" -ForegroundColor White
            
            if ([Math]::Abs($sizePercent) -lt 1) {
                Write-Host "   ✅ Output frames are nearly identical in size" -ForegroundColor Green
            } else {
                Write-Host "   ⚠️  Output frames differ in size - visual inspection recommended" -ForegroundColor Yellow
            }
        } else {
            Write-Host "   ⚠️  Could not find output frames for comparison" -ForegroundColor Yellow
        }
        
        Write-Host "`n📁 Output files saved to:" -ForegroundColor Yellow
        Write-Host "   • output_original/frame_0_original.jpg" -ForegroundColor White
        Write-Host "   • output_refactored/frame_0_refactored.jpg" -ForegroundColor White
        
        Write-Host "`n" -NoNewline
        Write-Host ("=" * 70) -ForegroundColor Cyan
    }
    
    default {
        Write-Host "❌ Invalid choice. Exiting." -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n✅ All tests complete!" -ForegroundColor Green

# Benchmark Runner for Functional Tests
# Runs all benchmarks and generates performance report

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Benchmark Suite Runner" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$TestDir = "functional-tests"
$Categories = @(
    "image-processing",
    "audio-processing",
    "parallel-processing",
    "performance"
)

$BenchmarkResults = @()

foreach ($Category in $Categories) {
    Write-Host ""
    Write-Host "Benchmarking: $Category" -ForegroundColor Yellow
    Write-Host "--------------------------------------"
    
    $TestPath = Join-Path $TestDir $Category
    
    if (Test-Path $TestPath) {
        # Run benchmarks
        $Output = go test -bench=. -benchmem "./$TestPath" 2>&1
        
        $Output | ForEach-Object {
            Write-Host $_
            
            # Parse benchmark results
            if ($_ -match "^Benchmark") {
                $BenchmarkResults += $_
            }
        }
    }
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Benchmark Summary" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

foreach ($Result in $BenchmarkResults) {
    Write-Host $Result
}

Write-Host ""
Write-Host "Benchmark run complete!" -ForegroundColor Green

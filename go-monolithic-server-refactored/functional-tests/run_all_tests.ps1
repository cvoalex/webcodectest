# Comprehensive Functional Test Runner
# Tests all optimization categories with detailed reporting

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Functional Test Suite Runner" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$TestDir = "functional-tests"
$Categories = @(
    "image-processing",
    "audio-processing",
    "parallel-processing",
    "integration",
    "performance",
    "edgecases"
)

$TotalTests = 0
$PassedTests = 0
$FailedTests = 0
$SkippedTests = 0

# Test each category
foreach ($Category in $Categories) {
    Write-Host ""
    Write-Host "Testing: $Category" -ForegroundColor Yellow
    Write-Host "--------------------------------------"
    
    $TestPath = Join-Path $TestDir $Category
    
    if (Test-Path $TestPath) {
        # Run tests with verbose output
        $Output = go test -v "./$TestPath" 2>&1
        
        # Parse results
        $Output | ForEach-Object {
            Write-Host $_
            
            if ($_ -match "^--- PASS:") {
                $PassedTests++
                $TotalTests++
            } elseif ($_ -match "^--- FAIL:") {
                $FailedTests++
                $TotalTests++
            } elseif ($_ -match "^--- SKIP:") {
                $SkippedTests++
            }
        }
        
        Write-Host ""
    } else {
        Write-Host "  [SKIP] Directory not found" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Test Summary" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Total Tests:   $TotalTests"
Write-Host "Passed:        $PassedTests" -ForegroundColor Green
Write-Host "Failed:        $FailedTests" $(if ($FailedTests -gt 0) { "-ForegroundColor Red" } else { "-ForegroundColor Green" })
Write-Host "Skipped:       $SkippedTests" -ForegroundColor Yellow
Write-Host ""

if ($FailedTests -eq 0) {
    Write-Host "All tests passed!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "Some tests failed!" -ForegroundColor Red
    exit 1
}

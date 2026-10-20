$ErrorActionPreference = 'Stop'
$testProject = $PSScriptRoot
$testPython = Join-Path $testProject '.venv\Scripts\python.exe'
$testNodeCandidates = @(
    (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe')
)
$testNodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
if ($testNodeCommand) { $testNodeCandidates += $testNodeCommand.Source }
$testNode = $testNodeCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (!(Test-Path -LiteralPath $testPython)) { throw 'Project Python environment is missing.' }
if (!$testNode) { throw 'Node.js is missing.' }
$savedTestUtf8 = $env:PYTHONUTF8
Push-Location $testProject
try {
    $env:PYTHONUTF8 = '1'
    & $testPython 'flight_delay_model\server\test_auth_history_api.py'
    $testBackendExit = $LASTEXITCODE
    Push-Location 'flight_delay_model\frontend'
    try {
        & $testNode --test 'tests\auth-state.test.mjs'
        $testFrontendExit = $LASTEXITCODE
    } finally { Pop-Location }
    if ($testBackendExit -ne 0 -or $testFrontendExit -ne 0) {
        throw 'Authentication/history checks failed. See .local/auth-history-*-test-results.json.'
    }
    Write-Host 'Authentication/history checks passed. Real account database was not modified.'
} finally {
    $env:PYTHONUTF8 = $savedTestUtf8
    Pop-Location
}

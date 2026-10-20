param(
    [ValidateSet('backend', 'frontend', 'build', 'check', 'weather')]
    [string]$Target = 'backend'
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$frontendDir = Join-Path $projectRoot 'flight_delay_model\frontend'
$previousPath = $env:PATH
$previousUtf8 = $env:PYTHONUTF8

function Assert-ExitCode([string]$Action) {
    if ($LASTEXITCODE -ne 0) { throw "$Action failed (exit $LASTEXITCODE)." }
}

Push-Location $projectRoot
try {
    $env:PYTHONUTF8 = '1'
    if ($Target -in @('frontend', 'build')) {
        $nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
        $candidates = @()
        if ($nodeCommand) { $candidates += $nodeCommand.Source }
        # Optional existing runtime on this computer; no global PATH edits.
        $candidates += Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
        $selectedNode = $null
        foreach ($candidate in $candidates) {
            if (!(Test-Path -LiteralPath $candidate)) { continue }
            $nodeVersion = [version]((& $candidate --version).Trim().TrimStart('v'))
            if (($nodeVersion.Major -eq 20 -and $nodeVersion -ge [version]'20.19.0') -or
                ($nodeVersion -ge [version]'22.12.0')) {
                $selectedNode = $candidate
                break
            }
        }
        if (!$selectedNode) { throw 'Install Node.js 24 LTS; current Node does not satisfy Vite.' }
        if (!(Test-Path (Join-Path $frontendDir 'node_modules\vite\bin\vite.js'))) {
            throw 'Frontend dependencies missing. Complete npm ci from the setup guide.'
        }
        $env:PATH = (Split-Path $selectedNode) + ';' + $env:PATH
        Set-Location $frontendDir
        Write-Host "Using Node $(& $selectedNode --version)"
        if ($Target -eq 'frontend') {
            & $selectedNode 'node_modules\vite\bin\vite.js' --host 127.0.0.1 --port 5173 --strictPort
        } else {
            & $selectedNode 'node_modules\vite\bin\vite.js' build
        }
        Assert-ExitCode 'Vite'
    } else {
        if (!(Test-Path -LiteralPath $projectPython)) { throw 'Create .venv first; see the setup guide.' }
        & $projectPython -c 'import flask, pandas, numpy, catboost, lightgbm'
        Assert-ExitCode 'Python dependency check'
        if (!(Test-Path 'flight_delay_model\server\flight_meta.db')) {
            throw 'Metadata database missing. Run scripts/prepare_local.py --apply first.'
        }
        switch ($Target) {
            'backend' { & $projectPython 'flight_delay_model\server\app.py' }
            'check' { & $projectPython 'flight_delay_model\server\test_backend.py' }
            'weather' { & $projectPython 'flight_delay_model\server\weather_serving.py' --self-test }
        }
        Assert-ExitCode $Target
    }
} finally {
    $env:PATH = $previousPath
    $env:PYTHONUTF8 = $previousUtf8
    Pop-Location
}

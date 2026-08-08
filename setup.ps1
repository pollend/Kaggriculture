[CmdletBinding()]
param(
    [switch] $SyncOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/"
    exit 127
}

Push-Location $repoRoot
try {
    Write-Host "Preparing the locked Python 3.12 environment..."
    & uv sync --locked --python 3.12
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    if ($SyncOnly) {
        Write-Host "Environment ready at $repoRoot/.venv"
        exit 0
    }

    $isWindowsPlatform = [System.IO.Path]::DirectorySeparatorChar -eq "\"
    if ($isWindowsPlatform) {
        $activate = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"
    }
    else {
        $activate = Join-Path $repoRoot ".venv/bin/Activate.ps1"
    }
    if (-not (Test-Path -LiteralPath $activate -PathType Leaf)) {
        Write-Error "uv did not create the expected activation script: $activate"
        exit 1
    }

    # Source Activate.ps1 inside the child process so its prompt, deactivate
    # function, PATH, and VIRTUAL_ENV all belong to the interactive session.
    $hostExecutable = (Get-Process -Id $PID).Path
    $escapedRoot = $repoRoot.Replace("'", "''")
    $escapedActivate = $activate.Replace("'", "''")
    $command = "Set-Location -LiteralPath '$escapedRoot'; . '$escapedActivate'; " +
        "Write-Host 'Environment active. Run exit to leave.'"

    Write-Host "Entering $repoRoot/.venv (exit to leave)..."
    & $hostExecutable -NoLogo -NoExit -Command $command
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

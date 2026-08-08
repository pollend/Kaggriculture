[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ArenaArgs
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
    & uv sync --python 3.12
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    if ($ArenaArgs.Count -gt 0 -and $ArenaArgs[0] -eq "setup") {
        Write-Host "Environment ready at $repoRoot/.venv"
        exit 0
    }

    if ($ArenaArgs.Count -eq 0) {
        $ArenaArgs = @(
            "match",
            "agents/configurable_crop_surrogate/main.py",
            "starter",
            "--seeds", "10"
        )
    }
    elseif ($ArenaArgs[0] -eq "competitors") {
        $extraArgs = @($ArenaArgs | Select-Object -Skip 1)
        $competitorAgents = @(
            Get-ChildItem -Path "agents/competitor_*/main.py" -File |
                Sort-Object FullName |
                ForEach-Object { $_.FullName }
        )
        if ($competitorAgents.Count -eq 0) {
            Write-Error "No competitor agents found under agents/competitor_*"
            exit 1
        }
        $ArenaArgs = @(
            "league",
            "agents/configurable_crop_surrogate/main.py"
        ) + $competitorAgents + @(
            "starter",
            "--seeds", "10"
        ) + $extraArgs
    }

    Write-Host "Running: python -m sim.arena $($ArenaArgs -join ' ')"
    & uv run --no-sync python -m sim.arena @ArenaArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

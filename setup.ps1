param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
Set-Location $ProjectRoot

Write-Host "[1/5] Checking Python..."
if (-not (Test-Path -LiteralPath $VenvPython)) {
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        & py.exe -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
        if ($LASTEXITCODE -ne 0) {
            throw "Python 3.11 or newer was not found. Install Python and run setup.bat again."
        }
        & py.exe -3 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv with py.exe." }
    }
    elseif (Get-Command python.exe -ErrorAction SilentlyContinue) {
        & python.exe -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
        if ($LASTEXITCODE -ne 0) {
            throw "Python 3.11 or newer was not found. Install Python and run setup.bat again."
        }
        & python.exe -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv with python.exe." }
    }
    else {
        throw "Python 3.11 or newer was not found. Install Python and run setup.bat again."
    }
}

Write-Host "[2/5] Checking packaging tools..."
& $VenvPython -m pip --version
if ($LASTEXITCODE -ne 0) { throw "pip is unavailable in the virtual environment." }

Write-Host "[3/5] Installing FundRebalance-Agent..."
& $VenvPython -m pip install --constraint requirements.lock -e ".[agent,app,dev]"
if ($LASTEXITCODE -ne 0) { throw "Failed to install project dependencies." }

Write-Host "[4/5] Creating local configuration..."
$DefaultProfile = Join-Path $ProjectRoot "user_data\default\profile.json"
if (-not (Test-Path -LiteralPath $DefaultProfile)) {
    & $VenvPython -m fund_agent.user_profile init --workspace "user_data\default" --name "default-user"
    if ($LASTEXITCODE -ne 0) { throw "Failed to initialize the default local profile." }
}

if (-not $SkipTests) {
    Write-Host "[5/5] Running tests..."
    & $VenvPython -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Tests failed. Review the output above." }
}
else {
    Write-Host "[5/5] Tests skipped."
}

Write-Host ""
Write-Host "Installation complete. Run .\start.bat to open the visual interface."

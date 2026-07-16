param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$UserData = Join-Path $ProjectRoot "user_data"
$PidFile = Join-Path $UserData ".streamlit.pid"
$StdoutLog = Join-Path $UserData "streamlit.stdout.log"
$StderrLog = Join-Path $UserData "streamlit.stderr.log"
$Url = "http://127.0.0.1:$Port"
Set-Location $ProjectRoot

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Virtual environment not found. Running setup first..."
    & (Join-Path $ProjectRoot "setup.ps1") -SkipTests
}

New-Item -ItemType Directory -Path $UserData -Force | Out-Null

$ExistingListener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($ExistingListener) {
    Write-Host "FundRebalance-Agent may already be running at $Url"
    Start-Process $Url
    exit 0
}

$Arguments = @(
    "-m", "streamlit", "run", "src\fund_agent\streamlit_app.py",
    "--server.port", "$Port",
    "--server.address", "127.0.0.1",
    "--server.headless", "true",
    "--browser.gatherUsageStats", "false"
)
$Process = Start-Process `
    -FilePath $VenvPython `
    -ArgumentList $Arguments `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -PassThru
Set-Content -LiteralPath $PidFile -Value $Process.Id -Encoding ascii

$Ready = $false
for ($Attempt = 0; $Attempt -lt 40; $Attempt++) {
    Start-Sleep -Milliseconds 500
    if ($Process.HasExited) {
        $ErrorText = Get-Content -LiteralPath $StderrLog -Raw -ErrorAction SilentlyContinue
        throw "Streamlit exited before startup. $ErrorText"
    }
    $Client = [System.Net.Sockets.TcpClient]::new()
    try {
        $Client.Connect("127.0.0.1", $Port)
        $Ready = $true
        break
    }
    catch {
    }
    finally {
        $Client.Dispose()
    }
}

if (-not $Ready) {
    throw "Streamlit did not become ready within 20 seconds. See $StderrLog"
}

Write-Host "FundRebalance-Agent is running at $Url"
Write-Host "Use .\stop.ps1 to stop it."
Start-Process $Url

$ErrorActionPreference = "Stop"
$PidFile = Join-Path $PSScriptRoot "user_data\.streamlit.pid"

if (-not (Test-Path -LiteralPath $PidFile)) {
    Write-Host "No FundRebalance-Agent PID file was found."
    exit 0
}

$ProcessId = [int](Get-Content -LiteralPath $PidFile -Raw)
$Process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
if ($Process) {
    $ExpectedPython = (Resolve-Path (Join-Path $PSScriptRoot ".venv\Scripts\python.exe")).Path
    $ProcessInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId"
    if (
        -not $ProcessInfo `
        -or $ProcessInfo.ExecutablePath -ne $ExpectedPython `
        -or $ProcessInfo.CommandLine -notmatch "streamlit"
    ) {
        throw "PID $ProcessId is not this project's Streamlit process; refusing to stop it."
    }
    Stop-Process -Id $ProcessId
    Write-Host "Stopped FundRebalance-Agent process $ProcessId."
}
else {
    Write-Host "The recorded process is no longer running."
}
Remove-Item -LiteralPath $PidFile -Force

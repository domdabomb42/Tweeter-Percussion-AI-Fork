param(
    [string]$FeedbackDir = (Join-Path $env:LOCALAPPDATA 'TweeterAIPercussion\feedback'),
    [string]$DpoRoot = '',
    [string]$PdExecutable = '',
    [int]$Port = 19785,
    [switch]$Cpu
)

$ErrorActionPreference = 'Stop'
$tweeterRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = (Resolve-Path (Join-Path $tweeterRoot '..\..')).Path
$modelRoot = Join-Path $repositoryRoot 'apps\smollm135\V4EA-synth'
$outputRoot = Join-Path $repositoryRoot 'apps\smollm135\MostRecentOutput'
if (-not $DpoRoot) { $DpoRoot = Join-Path $outputRoot 'DPOPreferenceRuns' }
$FeedbackDir = [IO.Path]::GetFullPath($FeedbackDir)
$DpoRoot = [IO.Path]::GetFullPath($DpoRoot)
New-Item -ItemType Directory -Force -Path $FeedbackDir | Out-Null
New-Item -ItemType Directory -Force -Path $DpoRoot | Out-Null
$python = (Get-Command python -ErrorAction Stop).Source
$stdout = Join-Path $FeedbackDir 'bridge.stdout.log'
$stderr = Join-Path $FeedbackDir 'bridge.stderr.log'

$running = $false
try {
    $client = [Net.Sockets.TcpClient]::new()
    $client.Connect('127.0.0.1', $Port)
    $client.Dispose()
    $running = $true
} catch { }

if (-not $running) {
    $arguments = @('-m', 'percussion_generation_v4ea', 'plugin-serve',
                   '--feedback-dir', ('"' + $FeedbackDir + '"'),
                   '--dpo-root', ('"' + $DpoRoot + '"'), '--port', "$Port")
    if ($Cpu) { $arguments += '--cpu' }
    $process = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $modelRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $deadline = [DateTime]::UtcNow.AddSeconds(120)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($process.HasExited) { throw "AI bridge exited with code $($process.ExitCode). See $stderr" }
        try {
            $client = [Net.Sockets.TcpClient]::new()
            $client.Connect('127.0.0.1', $Port)
            $client.Dispose()
            $running = $true
            break
        } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $running) { throw "AI bridge did not become ready. See $stderr" }
    Write-Host "AI bridge started (PID $($process.Id))."
} else {
    Write-Host "AI bridge is already listening on 127.0.0.1:$Port."
}

$patch = Join-Path $tweeterRoot 'L2Ork-Tweeter.pd'
if (-not $PdExecutable) {
    $found = Get-Command pd-l2ork,pd -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { $PdExecutable = $found.Source }
    if (-not $PdExecutable) {
        $installed = 'C:\Program Files (x86)\Pd-L2Ork\bin\pd-l2ork.exe'
        if (Test-Path -LiteralPath $installed) { $PdExecutable = $installed }
    }
}
if ($PdExecutable) {
    Start-Process -FilePath $PdExecutable -ArgumentList @('"' + $patch + '"') `
        -WorkingDirectory $tweeterRoot -WindowStyle Normal | Out-Null
    Write-Host 'Opened the Tweeter patch. Use RANDOM SESSION while disconnected from multiplayer.'
} else {
    Write-Host "Open this patch in PD-L2Ork: $patch"
}
Write-Host "Feedback directory: $FeedbackDir"
Write-Host "Bridge logs: $stdout and $stderr"

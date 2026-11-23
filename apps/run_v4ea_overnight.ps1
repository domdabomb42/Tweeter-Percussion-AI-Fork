param([switch]$DryRun)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$pythonCommand = (Get-Command python -ErrorAction Stop).Source

function Invoke-CheckedPython {
    param([string[]]$Arguments, [string]$Stage)
    Write-Host "[$Stage] python $($Arguments -join ' ')"
    if ($DryRun) { return }
    & $pythonCommand @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage failed with exit code $LASTEXITCODE"
    }
}

$models = @(
    @{ Name = '135MEA'; Folder = 'smollm135'; LoraRate = '1e-4'; EmbeddingRate = '1e-4'; Cpu = $false },
    @{ Name = '360MEA'; Folder = 'smollm360'; LoraRate = '1e-4'; EmbeddingRate = '5e-5'; Cpu = $false },
    @{ Name = '17BEA'; Folder = 'smollm1b'; LoraRate = '5e-5'; EmbeddingRate = '5e-5'; Cpu = $false }
)

foreach ($model in $models) {
    $modelRoot = Join-Path $PSScriptRoot $model.Folder
    $experiment = Join-Path $modelRoot 'V4EA-synth'
    $output = Join-Path $modelRoot 'MostRecentOutput\V4EA-positions-v1'
    $identity = Join-Path $output 'tokenizer_identity.json'
    $tokenizer = Join-Path $output 'tokenizer\tokenizer.json'
    $manifest = Join-Path $output 'manifest.json'
    $checkpointRoot = Join-Path $output 'checkpoints\rhythm_coverage_v1'
    $latest = Join-Path $checkpointRoot 'latest.json'

    Write-Host "Starting $($model.Name)"
    Push-Location $experiment
    try {
        if ((Test-Path -LiteralPath $identity) -ne (Test-Path -LiteralPath $tokenizer)) {
            throw "$($model.Name) has an incomplete tokenizer; inspect $output"
        }
        if (-not (Test-Path -LiteralPath $identity)) {
            Invoke-CheckedPython @('-m', 'percussion_generation_v4ea', '--output-dir', $output,
                'train-tokenizer', '--target-format', 'positions-v1', '--vocab-size', '16384') "$($model.Name) tokenizer"
        }
        if (-not (Test-Path -LiteralPath $manifest)) {
            Invoke-CheckedPython @('-m', 'percussion_generation_v4ea', '--output-dir', $output,
                'prepare', '--target-format', 'positions-v1') "$($model.Name) preparation"
        }
        Invoke-CheckedPython @('-m', 'percussion_generation_v4ea', '--output-dir', $output,
            'validate') "$($model.Name) dataset validation"

        $useCpu = [bool]$model.Cpu
        $alreadyFinished = $false
        $trainingArguments = @('-m', 'percussion_generation_v4ea', '--output-dir', $output,
            'train', '--dataset-passes', '2', '--examples-per-epoch', '536',
            '--lora-learning-rate', $model.LoraRate,
            '--embedding-learning-rate', $model.EmbeddingRate,
            '--save-steps', '0', '--patience', '3', '--progress-seconds', '180')
        if (Test-Path -LiteralPath $latest) {
            $pointer = Get-Content -LiteralPath $latest -Raw | ConvertFrom-Json
            $checkpoint = Join-Path $checkpointRoot $pointer.checkpoint
            $metadata = Get-Content -LiteralPath (Join-Path $checkpoint 'metadata.json') -Raw | ConvertFrom-Json
            $useCpu = $metadata.config.device_type -eq 'cpu'
            $trainingArguments += @('--resume-checkpoint', 'latest')
            $historyPath = Join-Path $checkpointRoot 'history.json'
            $finishedByPatience = $false
            if ((Test-Path -LiteralPath $historyPath) -and
                ($pointer.checkpoint -match '^(epoch_boundaries|final)/')) {
                $history = Get-Content -LiteralPath $historyPath -Raw | ConvertFrom-Json
                $finishedByPatience = (($history.PSObject.Properties.Name -contains 'early_stopped') -and
                    ($history.early_stopped -eq $true))
            }
            $alreadyFinished = ($metadata.epoch_index -ge $metadata.config.epochs) -or $finishedByPatience
        }
        if ($useCpu) { $trainingArguments += '--cpu' }
        if ($alreadyFinished) {
            Write-Host "$($model.Name) already finished; continuing."
        } elseif ($DryRun) {
            Invoke-CheckedPython $trainingArguments "$($model.Name) training"
            Write-Host "$($model.Name) planned; continuing."
        } else {
            Invoke-CheckedPython $trainingArguments "$($model.Name) training"
            Write-Host "$($model.Name) finished successfully; continuing."
        }
    }
    finally {
        Pop-Location
    }
}

if ($DryRun) {
    Write-Host 'Dry run finished; no training started.'
} else {
    Write-Host 'All three V4EA training stages completed.'
}

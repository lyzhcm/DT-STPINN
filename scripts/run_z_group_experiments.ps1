param(
    [ValidateSet("probe", "full", "benchmark")]
    [string]$Mode = "probe",

    [string[]]$Configs = @(
        "configs\ablation_z236_xml_reverse_odd_endpoint_time_until_rescue_0440_1900_floor.yaml",
        "configs\ablation_z237_xml_literal_endpoint_time_until_rescue_0440_1900_floor.yaml",
        "configs\ablation_z238_reverse_odd_no_timeuntil_shortcut_1900_floor.yaml",
        "configs\ablation_z239_reverse_odd_timeuntil_track_1900_floor.yaml",
        "configs\ablation_z240_reverse_odd_endpoint_eta_track_cold_1900_floor.yaml",
        "configs\ablation_z241_reverse_odd_endpoint_eta_high_track_1900_floor.yaml",
        "configs\ablation_z242_reverse_odd_endpoint_eta_high_track_no_sweep_1900_floor.yaml",
        "configs\ablation_z243_reverse_odd_constrained_sweep_rescue_1900_floor.yaml",
        "configs\ablation_z244_reverse_odd_constrained_sweep_rescue_relaxed_cold_body_1900_floor.yaml",
        "configs\ablation_z245_reverse_odd_constrained_sweep_rescue_relaxed_track_body_cold_1900_floor.yaml"
    ),

    [string]$VtuDir = "F:\VTU",
    [string]$Python = "python",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$GraphDevice = "cuda",

    [int]$ProbeEpochs = 10,
    [int]$ProbeTrainSamples = 300,
    [int]$ProbeValSamples = 80,
    [int]$ProbeTestSamples = 160,

    [int]$FullEpochs = 30,
    [int]$BenchmarkBatches = 20,

    [string]$SummaryPattern = "ablation_z2[3-4]*",
    [switch]$SkipPathDecision,
    [switch]$SkipSummary
)

$ErrorActionPreference = "Stop"

foreach ($Config in $Configs) {
    if (-not (Test-Path -LiteralPath $Config)) {
        throw "Config not found: $Config"
    }
    if (-not (Test-Path -LiteralPath $VtuDir)) {
        throw "VTU directory not found: $VtuDir"
    }

    $stem = [System.IO.Path]::GetFileNameWithoutExtension($Config)
    $experimentName = switch ($Mode) {
        "probe" { "${stem}_probe${ProbeTrainSamples}" }
        "full" { "${stem}_full" }
        "benchmark" { "${stem}_bench${BenchmarkBatches}" }
    }

    $trainArgs = @(
        "scripts\train.py",
        "--config", $Config,
        "--vtu_dir", $VtuDir,
        "--graph_device", $GraphDevice,
        "--experiment_name", $experimentName
    )

    if ($Mode -eq "probe") {
        $trainArgs += @(
            "--epochs", $ProbeEpochs,
            "--max_train_samples", $ProbeTrainSamples,
            "--max_val_samples", $ProbeValSamples,
            "--max_test_samples", $ProbeTestSamples
        )
    } elseif ($Mode -eq "full") {
        $trainArgs += @("--epochs", $FullEpochs)
    } elseif ($Mode -eq "benchmark") {
        $trainArgs += @("--benchmark_batches", $BenchmarkBatches, "--skip_test")
    }

    Write-Host ""
    Write-Host "=== Running ${Mode}: $experimentName ==="
    Write-Host "$Python $($trainArgs -join ' ')"
    & $Python @trainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Training command failed for $experimentName with exit code $LASTEXITCODE"
    }
}

if (-not $SkipSummary -and $Mode -ne "benchmark") {
    $csv = "results\z_experiment_summary.csv"
    $md = "results\z_experiment_summary.md"
    Write-Host ""
    Write-Host "=== Summarizing z-group runs ==="
    & $Python "scripts\summarize_z_experiments.py" `
        "--logs_dir" "logs" `
        "--pattern" $SummaryPattern `
        "--output_csv" $csv `
        "--output_md" $md
    if ($LASTEXITCODE -ne 0) {
        throw "Summary command failed with exit code $LASTEXITCODE"
    }

    $hasPathPair = ($Configs | Where-Object { $_ -like "*z236*" }).Count -gt 0 -and `
        ($Configs | Where-Object { $_ -like "*z237*" }).Count -gt 0
    if (-not $SkipPathDecision -and $hasPathPair) {
        Write-Host ""
        Write-Host "=== Deciding z236/z237 path alignment ==="
        & $Python "scripts\decide_z236_z237.py" `
            "--summary_csv" $csv `
            "--output_md" "results\z236_z237_decision.md"
        if ($LASTEXITCODE -ne 0) {
            throw "Decision command failed with exit code $LASTEXITCODE"
        }
    }
}

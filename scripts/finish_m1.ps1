# Finishes M1 end to end: trains the missing folds, exports all four, validates, runs Gate 2,
# builds the real backend bundle. Resumable: re-run the same command after any interruption;
# finished folds are skipped (same config hash + checkpoint present).
#
# Run from the Prototype folder with the .venv active:
#     .\scripts\finish_m1.ps1
# Optional: .\scripts\finish_m1.ps1 -SkipTrain   (only export onward)
#
# Before starting: plug the laptop in and stop it sleeping (Settings > System > Power > Never),
# otherwise Windows suspends mid-fold, which is very likely what killed the fold 2019 run.

param([switch]$SkipTrain)

$P = "C:\Users\ad\PycharmProjects\WildFireDetection\Prototype"
$WSTS = "C:\Users\ad\PycharmProjects\WildFireDetection\WildfireSpreadTS\WildfireSpreadTS"
$RUN = "m1_spatial_unet_T1_seed0"
$RUNID = "m1_t1_v1"
Set-Location $P
$log = "$P\outputs\finish_m1_$(Get-Date -Format yyyyMMdd_HHmmss).log"
New-Item -ItemType Directory -Force "$P\outputs" | Out-Null

function Step($name, [scriptblock]$cmd) {
    Write-Host "`n=== $name  ($(Get-Date -Format HH:mm))" -ForegroundColor Cyan
    "=== $name $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
    & $cmd 2>&1 | Tee-Object -Append -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED at: $name  (full output in $log). Fix, then re-run this script; it resumes." -ForegroundColor Red
        exit 1
    }
}

Step "0/7 environment check" {
    python -c "import torch, fireguard_ml; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
}

if (-not $SkipTrain) {
    Step "1/7 train M1, all folds (fold 2018 is skipped; about 1-2 h per remaining fold)" {
        python scripts\train_temporal.py --model m1_spatial_unet --source wsts --data-root $WSTS --out outputs\m1 --preprocessing-version wsts_strict_input_v1 --device cuda --resume
    }
}

foreach ($f in 2018, 2019, 2020, 2021) {
    if (-not (Test-Path "$P\outputs\m1\fold_$f\$RUN.pt")) {
        Write-Host "Missing checkpoint for fold $f; re-run without -SkipTrain." -ForegroundColor Red; exit 1
    }
}

Step "2/7 export held-out forecasts, all four folds (slow: writes one GeoTIFF per test sample)" {
    python scripts\export_temporal_oof.py --run-dir outputs\m1 --run $RUN --source wsts --data-root $WSTS --artifact-root artifacts --run-id $RUNID --device cuda
}
Step "3/7 replay catalogue (expect 0 rejected)" {
    python scripts\build_replay_catalogue.py --artifact-root artifacts --run-id $RUNID
}
Step "4/7 Role 4 validation + reveal package (expect coverage 1.0, 0 metric mismatches)" {
    python scripts\validate_oof_and_build_reveal.py --artifact-root artifacts --run-id $RUNID --data-root $WSTS
}
Step "5/7 Gate 2 comparison (prints BEATS / MATCHES / FAILS)" {
    python scripts\run_model_comparison.py --model m1_unet=artifacts\runs\$RUNID\post_reveal\per_event_test_scores.csv
}
Step "6/7 backend registry bundle (expect mode: real)" {
    python scripts\build_backend_registry.py --artifact-root artifacts --run-id $RUNID --data-root $WSTS --backend-artifact-root backend\artifacts --out artifacts\registry\$RUNID
}

Write-Host "`n=== 7/7 summary" -ForegroundColor Cyan
foreach ($f in 2018, 2019, 2020, 2021) {
    $s = Get-Content "$P\outputs\m1\fold_$f\$RUN.summary.json" | ConvertFrom-Json
    "fold {0}: best_epoch {1}, val AUPRC {2:N4}, {3:N0} min" -f $f, $s.best_epoch, $s.val_auprc, ($s.seconds / 60)
}
Write-Host "`nDone. Gate 2 verdict is in reports\phase3\model_comparison.md and gate_decisions.json." -ForegroundColor Green
Write-Host "Next: paste it into docs\phase_gate_log.md (Gate 2 row) for the lead; then RUNBOOK section 6 to view it in the site." -ForegroundColor Green
Write-Host "Full log: $log"

# Builds the demo data: fold-2018 M1 replays (about 220 real 2018 fires, model never saw them)
# plus the synthetic live-forecast smoke run, then starts the backend on it.
# Run from the Prototype folder with the .venv active:   .\scripts\demo_fold2018.ps1
# Writes only to artifacts\runs\m1_fold2018_preview and prototype_work\ (both git-ignored).

$P = "C:\Users\ad\PycharmProjects\WildFireDetection\Prototype"
$WSTS = "C:\Users\ad\PycharmProjects\WildFireDetection\WildfireSpreadTS\WildfireSpreadTS"
$RUN = "m1_fold2018_preview"
$OUT = "$P\prototype_work\fold2018"
Set-Location $P

function Step($name, [scriptblock]$cmd) {
    Write-Host "`n=== $name" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at: $name" -ForegroundColor Red; exit 1 }
}

Step "1/6 export fold 2018 forecasts (slowest step)" {
    python scripts\export_temporal_oof.py --run-dir outputs\m1 --run m1_spatial_unet_T1_seed0 --folds 2018 --source wsts --data-root $WSTS --artifact-root artifacts --run-id $RUN --device cuda
}
Step "2/6 replay catalogue" {
    python scripts\build_replay_catalogue.py --artifact-root artifacts --run-id $RUN
}
Step "3/6 Role 4 validation + reveal package" {
    python scripts\validate_oof_and_build_reveal.py --artifact-root artifacts --run-id $RUN --data-root $WSTS --folds 2018
}
Step "4/6 backend registry bundle (expect mode: mock, only fold 2018)" {
    python scripts\build_backend_registry.py --artifact-root artifacts --run-id $RUN --data-root $WSTS --backend-artifact-root "$OUT\backend_artifacts" --out "$OUT\registry"
}
Step "5/6 live forecast smoke run (synthetic checkpoints, CPU)" {
    python scripts\smoke_live_forecast.py --work prototype_work\live_smoke
}
Copy-Item -Recurse -Force "$P\prototype_work\live_smoke\backend_artifacts\live" "$OUT\backend_artifacts\"

Write-Host "`n=== 6/6 load database and start backend" -ForegroundColor Cyan
Set-Location "$P\backend"
$env:FIREGUARD_APP_ENV = "development"
$env:FIREGUARD_ARTIFACT_ROOT = "$OUT\backend_artifacts"
$env:DATABASE_URL = "sqlite:///$($OUT -replace '\\','/')/fireguard_demo.db"
python -m alembic upgrade head
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at: alembic" -ForegroundColor Red; exit 1 }
python scripts\load_registry_bundle.py --bundle "$OUT\registry\registry_bundle.json"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at: load bundle" -ForegroundColor Red; exit 1 }
Write-Host "`nBackend starting. In a second window: cd $P\frontend ; npm run dev" -ForegroundColor Green
python -m uvicorn fireguard.main:app --port 8000

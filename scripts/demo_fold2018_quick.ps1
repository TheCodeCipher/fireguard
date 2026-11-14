# Quick preview: the real fold-2018 M1 model on the five largest 2018 test fires (chosen by
# fire size, not by score). The run is marked mock (coverage < 1, one fold). A few minutes on GPU.
# Run from the Prototype folder with the .venv active:
#   powershell -ExecutionPolicy Bypass -File scripts\demo_fold2018_quick.ps1

$P = "C:\Users\ad\PycharmProjects\WildFireDetection\Prototype"
$WSTS = "C:\Users\ad\PycharmProjects\WildFireDetection\WildfireSpreadTS\WildfireSpreadTS"
$RUN = "m1_fold2018_preview"
$OUT = "$P\prototype_work\fold2018"
# Use the project venv's python explicitly and put ml\src first on the path, so an older
# installed copy of fireguard_ml elsewhere can never be imported instead of this checkout.
$PY = "C:\Users\ad\PycharmProjects\WildFireDetection\.venv\Scripts\python.exe"
$env:PYTHONPATH = "$P\ml\src;$P\backend\src"
& $PY -c "import fireguard_ml.inference.temporal_oof as m, inspect; print('using', m.__file__); assert 'events' in inspect.signature(m.export_oof).parameters, 'old fireguard_ml copy'"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: wrong fireguard_ml copy (see 'using' line)" -ForegroundColor Red; exit 1 }
$EVENTS = @("fire_24332628", "fire_25294714", "fire_25295099", "fire_21889754", "fire_24461623")
Set-Location $P
if (Test-Path "$P\artifacts\runs\$RUN") { Remove-Item -Recurse -Force "$P\artifacts\runs\$RUN" }
if (Test-Path $OUT) { Remove-Item -Recurse -Force $OUT }

function Step($name, [scriptblock]$cmd) {
    Write-Host "`n=== $name" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at: $name" -ForegroundColor Red; exit 1 }
}

Step "1/5 export 5 fires with the fold-2018 model" {
    & $PY scripts\export_temporal_oof.py --run-dir outputs\m1 --run m1_spatial_unet_T1_seed0 --folds 2018 --events $EVENTS --source wsts --data-root $WSTS --artifact-root artifacts --run-id $RUN --device cuda
}
Step "2/5 replay catalogue" {
    & $PY scripts\build_replay_catalogue.py --artifact-root artifacts --run-id $RUN
}
Step "3/5 Role 4 validation + reveal package" {
    & $PY scripts\validate_oof_and_build_reveal.py --artifact-root artifacts --run-id $RUN --data-root $WSTS --folds 2018
}
Step "4/5 registry bundle (expect mode: mock)" {
    & $PY scripts\build_backend_registry.py --artifact-root artifacts --run-id $RUN --data-root $WSTS --backend-artifact-root "$OUT\backend_artifacts" --out "$OUT\registry"
}
if (Test-Path "$P\prototype_work\real_smoke\backend_artifacts\live") {
    Copy-Item -Recurse -Force "$P\prototype_work\real_smoke\backend_artifacts\live" "$OUT\backend_artifacts\"
}

Write-Host "`n=== 5/5 load database and start backend (stop the old uvicorn first)" -ForegroundColor Cyan
Set-Location "$P\backend"
$env:FIREGUARD_APP_ENV = "development"
$env:FIREGUARD_ARTIFACT_ROOT = "$OUT\backend_artifacts"
$env:DATABASE_URL = "sqlite:///$($OUT -replace '\\','/')/fireguard_demo.db"
& $PY -m alembic upgrade head
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at: alembic" -ForegroundColor Red; exit 1 }
& $PY scripts\load_registry_bundle.py --bundle "$OUT\registry\registry_bundle.json"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at: load bundle" -ForegroundColor Red; exit 1 }
Write-Host "`nBackend starting on the fold-2018 preview. Ctrl+F5 the site." -ForegroundColor Green
& $PY -m uvicorn fireguard.main:app --port 8000

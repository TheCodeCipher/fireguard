# Cleans the WildFireDetection workspace down to the integrated prototype.
#
#   cd C:\Users\ad\PycharmProjects\WildFireDetection\Prototype
#   .\.venv\Scripts\Activate.ps1          (or whichever venv you use; needed only for part 2)
#   .\scripts\cleanup_workspace.ps1
#
# Part 1  deletes the old / un-integrated copies next to Prototype:
#           Archive, branches, fireguard, role-1, role-4
#         keeps: Prototype, WildfireSpreadTS (dataset), .venv, .idea
#         and removes frontend\src\mscf (the "simulate output" module; nothing imports it).
# Part 2  strips "Claude" from the git history of integration/prototype only
#         (author of commit 396f99f + Co-Authored-By / Claude-Session trailers).
#         It rewrites LOCALLY and keeps a backup branch; it never pushes. The push command
#         is printed at the end for you to run when you are ready.
#
# Nothing in the working-tree files mentions Claude; the only mentions are in git metadata.

param([switch]$SkipDelete, [switch]$SkipHistory)

$ErrorActionPreference = "Stop"
$P = "C:\Users\ad\PycharmProjects\WildFireDetection\Prototype"
$ROOT = Split-Path $P -Parent
Set-Location $P

# ---------------------------------------------------------------- part 1: files
if (-not $SkipDelete) {
    $targets = "Archive", "branches", "fireguard", "role-1", "role-4" |
        ForEach-Object { Join-Path $ROOT $_ } | Where-Object { Test-Path $_ }
    Write-Host "`nWill permanently delete:" -ForegroundColor Yellow
    $targets | ForEach-Object { Write-Host "  $_" }
    Write-Host "  $P\frontend\src\mscf"
    Write-Host "Keeping: Prototype, WildfireSpreadTS, .venv, .idea" -ForegroundColor Green
    if ((Read-Host "Type DELETE to continue") -ne "DELETE") { Write-Host "Skipped deletion."; }
    else {
        foreach ($t in $targets) {
            Write-Host "deleting $t"
            # cmd's rmdir copes with read-only .git files and long node_modules paths better
            cmd /c "rmdir /s /q `"$t`""
            if (Test-Path $t) { Write-Host "  could not fully delete $t (file open in PyCharm/Explorer?)" -ForegroundColor Red }
        }
        git rm -r -q --ignore-unmatch frontend/src/mscf
        if (Test-Path "frontend\src\mscf") { Remove-Item -Recurse -Force "frontend\src\mscf" }
        git add -A frontend/src scripts/train_temporal.py scripts/finish_m1.ps1 scripts/cleanup_workspace.ps1
        git commit -q -m "chore: remove unused simulate-output module; fix fold loop crash; add finish_m1 and cleanup scripts"
        Write-Host "Part 1 done." -ForegroundColor Green
    }
}

# ---------------------------------------------------------------- part 2: git history
if (-not $SkipHistory) {
    $dirty = git status --porcelain
    if ($dirty) { Write-Host "Working tree has uncommitted changes; commit or stash them, then re-run with -SkipDelete." -ForegroundColor Red; exit 1 }
    python -m pip install -q git-filter-repo
    git branch -f backup/before-history-rewrite HEAD
    $tmp = New-Item -ItemType Directory -Force "$env:TEMP\fg_rewrite"
    "Arkodyuti Bhattacharyya <ark0b@users.noreply.github.com> Claude <noreply@anthropic.com>" |
        Out-File -Encoding ascii "$tmp\mailmap.txt"
    @(
        'regex:(?im)^(co-authored-by:[ \t]*claude[^\n]*|claude-session:[^\n]*)\n?==>',
        'regex:\n{3,}==>\n\n'
    ) | Out-File -Encoding ascii "$tmp\messages.txt"
    # Only commits that are on integration/prototype but not on main are rewritten; main is untouched.
    python -m git_filter_repo --force --refs origin/main..integration/prototype `
        --mailmap "$tmp\mailmap.txt" --replace-message "$tmp\messages.txt"
    if ($LASTEXITCODE -ne 0) { Write-Host "Rewrite failed; nothing changed (backup branch: backup/before-history-rewrite)." -ForegroundColor Red; exit 1 }

    $left = git log origin/main..integration/prototype --format="%an %ae%n%B" | Select-String -Pattern "claude|anthropic"
    if ($left) { Write-Host "Still found:" -ForegroundColor Red; $left } else { Write-Host "integration/prototype history: no Claude mentions left." -ForegroundColor Green }
    Write-Host "`nLocal rewrite done. Backup of the old history: branch backup/before-history-rewrite." -ForegroundColor Green
    Write-Host "To update GitHub (rewrites the shared branch; teammates who pulled it must re-clone or reset):" -ForegroundColor Yellow
    Write-Host "    git push --force-with-lease origin integration/prototype"
    Write-Host "One older mention stays on main (squash-merge #18). Changing it needs a force-push of main: the lead's call." -ForegroundColor Yellow
}

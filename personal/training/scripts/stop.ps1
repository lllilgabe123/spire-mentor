# Stop the Spire Mentor run-AI training. Progress up to the last finished round
# is kept; "Start AI training" carries on from there.
$Root = Split-Path -Parent $PSScriptRoot
$cfg = Get-Content (Join-Path $Root 'campaign.json') -Raw | ConvertFrom-Json
$camp = Join-Path $Root "campaign\$($cfg.name)"
$pidFile = Join-Path $camp 'trainer.pid'
$id = if (Test-Path $pidFile) { [int](Get-Content $pidFile) } else { 0 }
if ($id -and (Get-Process -Id $id -ErrorAction SilentlyContinue)) {
    New-Item (Join-Path $camp 'stop-requested') -ItemType File -Force | Out-Null
    & taskkill.exe /PID $id /T /F | Out-Null
    Remove-Item $pidFile -ErrorAction SilentlyContinue
    $n = Get-ChildItem $camp -Filter 'main.iter_*.pt' | ForEach-Object { [int]($_.BaseName -replace '^main\.iter_', '') } | Measure-Object -Maximum
    Write-Host ("Training stopped. {0} of {1} rounds are saved; click Start AI training to carry on." -f [int]$n.Maximum, $cfg.iterations)
} else {
    Write-Host 'Training is not running.'
}
Start-Sleep 4

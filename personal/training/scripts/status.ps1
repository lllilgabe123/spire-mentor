# Show how the Spire Mentor run-AI training is going.
$Root = Split-Path -Parent $PSScriptRoot
$cfg = Get-Content (Join-Path $Root 'campaign.json') -Raw | ConvertFrom-Json
$camp = Join-Path $Root "campaign\$($cfg.name)"
$save = Join-Path $camp 'main'
$pidFile = Join-Path $camp 'trainer.pid'
$host.UI.RawUI.WindowTitle = 'Spire Mentor AI training status'

$running = (Test-Path $pidFile) -and (Get-Process -Id ([int](Get-Content $pidFile)) -ErrorAction SilentlyContinue)
$n = Get-ChildItem $camp -Filter 'main.iter_*.pt' -ErrorAction SilentlyContinue |
    ForEach-Object { [int]($_.BaseName -replace '^main\.iter_', '') } | Measure-Object -Maximum
$done = if ($n.Count) { [int]$n.Maximum } else { 0 }

# One row per round (a resumed round can appear twice; keep its last row).
$rows = @{}
if (Test-Path "$save.stats.jsonl") {
    Get-Content "$save.stats.jsonl" | Where-Object { $_.Trim() } | ForEach-Object {
        $r = $_ | ConvertFrom-Json
        if ($r.iteration -lt $done) { $rows[[int]$r.iteration] = $r }
    }
}
$list = $rows.Keys | Sort-Object | ForEach-Object { $rows[$_] }

Write-Host ''
Write-Host ("Spire Mentor AI training: {0}, beta {1}" -f $cfg.character, $cfg.game_version)
Write-Host ("  Status:  {0}" -f $(if ($running) { 'RUNNING' } else { 'stopped' }))
Write-Host ("  Rounds:  {0} of {1} saved ({2:P0})" -f $done, $cfg.iterations, ($done / $cfg.iterations))
if ($list.Count) {
    $mins = ($list | Select-Object -Last 10 | Measure-Object iter_s -Average).Average / 60
    $left = ($cfg.iterations - $done) * $mins / 60
    Write-Host ("  Speed:   about {0:N0} minutes per round; about {1:N1} hours of training left" -f $mins, $left)
    function WinOver($subset, $lo, $hi) {
        $w = 0.0; $g = 0
        foreach ($r in $subset) { for ($a = $lo; $a -le $hi; $a++) {
            $k = "n_runs/A$a"; $c = $r.$k
            if ($c) { $g += $c; $w += $c * $r."win_rate/A$a" } } }
        if ($g) { '{0:P0} of {1} games' -f ($w / $g), $g } else { 'no games yet' }
    }
    $first = $list | Select-Object -First 10
    $last = $list | Select-Object -Last 10
    Write-Host ''
    Write-Host '  Wins during training (first 10 rounds -> latest 10 rounds):'
    Write-Host ('    All ascensions:  {0}  ->  {1}' -f (WinOver $first 0 10), (WinOver $last 0 10))
    Write-Host ('    Ascension 8-10:  {0}  ->  {1}' -f (WinOver $first 8 10), (WinOver $last 8 10))
    Write-Host ''
    Write-Host '  These are practice numbers (the AI explores while it learns). The real'
    Write-Host '  test comes at the end: new AI vs current AI on the same seeds.'
}
Write-Host ''
Read-Host 'Press Enter to close'

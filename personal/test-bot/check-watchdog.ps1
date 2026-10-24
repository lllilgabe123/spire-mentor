<#
  check-watchdog.ps1 -- prove Spire Mentor's in-game watchdog fires.
  Swaps the ORIGINAL AI engine back in (it cannot read Midnight), runs a
  short co-op test where the host gives itself Midnight, then ALWAYS
  restores the fixed engine and reports what the watchdog recorded.
#>
$engine = 'E:\Steam\steamapps\common\Slay the Spire 2\mods\SpireMentor\SimAdvisor\_internal\sts2_rs\sts2_rs.pyd'
$original = 'E:\Steam\steamapps\common\Slay the Spire 2\mods_backup\sts2_rs.pyd.original'
$fixed = Join-Path $env:LOCALAPPDATA 'sts2t\wheel-x\sts2_rs\sts2_rs.pyd'
$wd = Join-Path $env:LOCALAPPDATA 'SpireMentor\watchdog.jsonl'
$before = if (Test-Path $wd) { @(Get-Content $wd).Count } else { 0 }
$start = Get-Date
try {
    Copy-Item $original $engine -Force
    & (Join-Path $PSScriptRoot 'run-mp-test.ps1') -Mode play -Floors 3 -TimeoutMinutes 12 -CoopTrainer -CoopTrainerCard MIDNIGHT *>&1 |
        Select-String -Pattern 'Results:|Duration|FAIL|Host received MIDNIGHT|floor reached' | ForEach-Object { "$_" }
}
finally {
    Get-Process SlayTheSpire2 -ErrorAction SilentlyContinue | Where-Object { $_.StartTime -ge $start } | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep 3
    Copy-Item $fixed $engine -Force
    "fixed engine restored: $((Get-FileHash $engine).Hash -eq (Get-FileHash $fixed).Hash)"
}
$new = if (Test-Path $wd) { @(Get-Content $wd | Select-Object -Skip $before) } else { @() }
"new watchdog records: $($new.Count)"
$new | Select-Object -First 3
$r = Get-ChildItem (Join-Path $PSScriptRoot 'results') -Directory | Sort-Object Name | Select-Object -Last 1
"in-game warning lines in the host log:"
Select-String -Path (Join-Path $r.FullName 'host-godot.log') -Pattern 'SimWatchdog' | Select-Object -First 3 | ForEach-Object { '  ' + $_.Line }
"run note written: " + [bool](Select-String -Path (Join-Path $env:LOCALAPPDATA 'SpireMentor\runs.jsonl') -Pattern 'Host' -Quiet -ErrorAction SilentlyContinue)

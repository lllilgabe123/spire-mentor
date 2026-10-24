<#
  run-stress.ps1 -- several long two-player runs in a row (run-mp-test.ps1),
  then one combined report: results\stress-<timestamp>.txt
#>
param(
    [int]$Floors = 20,
    [int]$TimeoutMinutes = 60
)
$here = $PSScriptRoot
$test = Join-Path $here 'run-mp-test.ps1'
$runs = @(
    @{ name = 'plain'; args = @{} },
    @{ name = 'host gives itself Midnight'; args = @{ CoopTrainer = $true; CoopTrainerCard = 'MIDNIGHT' } },
    @{ name = 'host gives gifts to the friend'; args = @{ CoopTrainer = $true; CoopTrainerOther = $true } }
)
$report = Join-Path $here ("results\stress-{0}.txt" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Force (Split-Path $report) | Out-Null
foreach ($run in $runs) {
    while (Get-Process SlayTheSpire2 -ErrorAction SilentlyContinue) { Start-Sleep 5 }
    "===== $($run.name) =====" | Add-Content $report -Encoding utf8
    $a = $run.args
    try {
        & $test -Mode play -Floors $Floors -TimeoutMinutes $TimeoutMinutes @a *>&1 |
            ForEach-Object { "$_" } | Add-Content $report -Encoding utf8
    }
    catch {
        "SCRIPT ERROR: $_" | Add-Content $report -Encoding utf8
    }
    '' | Add-Content $report -Encoding utf8
}
"REPORT: $report"
Get-Content $report | Select-String -Pattern '^=====|FAIL|PASS  Reached|floor reached|Duration|MISMATCH|SCRIPT ERROR|stuck|error'

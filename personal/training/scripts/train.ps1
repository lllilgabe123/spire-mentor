# Start (or resume) the Spire Mentor run-AI training campaign.
#
# Double-click "Start AI training" to run this. It:
#   * refuses while Slay the Spire 2 is open (training uses every core);
#   * resumes from the newest saved round, or warm-starts from the shipped AI;
#   * runs the trainer at below-normal priority, so the PC stays usable;
#   * keeps the PC awake while this window is open (no system setting changes:
#     the request ends when the window closes);
#   * stops training by itself if Slay the Spire 2 is started, so the game
#     never lags. Click Start again afterwards to carry on.
# Nothing here touches the game or its mods.

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$cfg = Get-Content (Join-Path $Root 'campaign.json') -Raw | ConvertFrom-Json
$camp = Join-Path $Root "campaign\$($cfg.name)"
$save = Join-Path $camp 'main'
$pidFile = Join-Path $camp 'trainer.pid'
$py = Join-Path $Root 'src\sts2-pycli\.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force $camp | Out-Null
$host.UI.RawUI.WindowTitle = 'Spire Mentor AI training'

function Say($msg) { Write-Host ("[{0:HH:mm}] {1}" -f (Get-Date), $msg) }
function GameRunning { [bool](Get-Process SlayTheSpire2 -ErrorAction SilentlyContinue) }
function TrainerAlive {
    if (-not (Test-Path $pidFile)) { return $null }
    $id = [int](Get-Content $pidFile)
    Get-Process -Id $id -ErrorAction SilentlyContinue
}
# Rounds done = the newest saved round (main.iter_N.pt), not stats rows: a stop
# between a round's stats line and its save repeats that round on resume.
function RoundsDone {
    $n = Get-ChildItem $camp -Filter 'main.iter_*.pt' -ErrorAction SilentlyContinue |
        ForEach-Object { [int]($_.BaseName -replace '^main\.iter_', '') } | Measure-Object -Maximum
    if ($n.Count) { [int]$n.Maximum } else { 0 }
}

if (TrainerAlive) { Say 'Training is already running. Use "AI training status" to check on it.'; Read-Host 'Press Enter to close'; exit 0 }
if (GameRunning) { Say 'Slay the Spire 2 is open. Close it first: training uses every processor core.'; Read-Host 'Press Enter to close'; exit 1 }

# Newest checkpoint that actually loads (a stop during a save can leave a
# half-written newest file; fall back to the one before it).
$resume = 0
$ckpts = Get-ChildItem $camp -Filter 'main.iter_*.pt' -ErrorAction SilentlyContinue |
    ForEach-Object { [pscustomobject]@{ N = [int]($_.BaseName -replace '^main\.iter_', ''); F = $_.FullName } } |
    Sort-Object N -Descending
foreach ($c in $ckpts) {
    & $py -c "import sys, torch; torch.load(sys.argv[1], map_location='cpu', weights_only=True)" $c.F 2>$null
    if ($LASTEXITCODE -eq 0) { $resume = $c.N; break }
    Say "Saved round $($c.N) is incomplete (stopped mid-save); using the one before it."
    Rename-Item $c.F "$($c.F).broken" -Force
}
if ($resume -ge $cfg.iterations) { Say "Training is finished ($resume of $($cfg.iterations) rounds). Ask Claude to test the new AI."; Read-Host 'Press Enter to close'; exit 0 }

# "cuda": the learner update runs on the graphics card via gpu_trainer.py
# (workers stay on the small CPU torch; see that file). "cpu": plain trainer.
$entry = if ($cfg.device -eq 'cuda') { @((Join-Path $PSScriptRoot 'gpu_trainer.py')) } else { @('-m', 'ppo.trainer') }
$targs = @($entry) + @(
    '--character', $cfg.character,
    '--iterations', $cfg.iterations,
    '--games-per-iter', $cfg.games_per_iter,
    '--procs', $cfg.procs,
    '--sims', $cfg.sims, '--boss-sims', $cfg.boss_sims,
    '--ascension-weights', $cfg.ascension_weights,
    '--act1', 'roll',
    '--entropy-coef', $cfg.entropy_coef, '--entropy-coef-final', $cfg.entropy_coef,
    '--lr-anneal-final', $cfg.lr_anneal_final,
    '--save-path', $save, '--save-every', '1',
    '--device', $cfg.device, '--allow-plain-wheel')
if ($resume -gt 0) { $targs += @('--resume-from-iter', $resume); Say "Resuming after round $resume of $($cfg.iterations)." }
else { $targs += @('--init-ckpt', (Join-Path $Root $cfg.init_ckpt)); Say "Starting fresh from the current shipped AI ($($cfg.iterations) rounds planned)." }

$env:STS2_GAME_VERSION = $cfg.game_version
# Fights the run AI trains against: "clairvoyant" matches the live advisor's
# --fight-sight clairvoyant (STS2_RUNAI_HONEST=0); anything else keeps the
# deployed honest search.
if ($cfg.fight_sight -eq 'clairvoyant') { $env:STS2_RUNAI_HONEST = '0'; Say 'Fights: see-the-future (like your game).' }
else { Remove-Item Env:STS2_RUNAI_HONEST -ErrorAction SilentlyContinue }
$env:PYTHONPATH = 'python'
$env:PYTHONUTF8 = '1'
# Below-normal priority is inherited by the trainer and all of its workers.
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$p = Start-Process $py -ArgumentList $targs -WorkingDirectory (Join-Path $Root 'src\sts2-pycli') `
    -RedirectStandardOutput (Join-Path $camp "log-$stamp.txt") -RedirectStandardError (Join-Path $camp "log-$stamp.err.txt") `
    -WindowStyle Hidden -PassThru
Set-Content $pidFile $p.Id

# Keep the PC awake while this window is open (ES_CONTINUOUS | ES_SYSTEM_REQUIRED).
# Declared with an int parameter: PowerShell 5.1 reads 0x80000001 as a negative
# Int32 and refuses to convert it to uint; as int the bits are the same.
Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(int f);'
try { [Win.Power]::SetThreadExecutionState(0x80000001) | Out-Null } catch { Say "Could not keep the PC awake: $_" }

Say 'Training is running. Leave this window open (minimise it) to keep the PC awake.'
Say 'To stop: double-click "Stop AI training" (or just start Slay the Spire 2).'
$seen = RoundsDone
while ($true) {
    Start-Sleep 20
    if ($p.HasExited) {
        $done = RoundsDone
        if ($done -ge $cfg.iterations) { Say "Training finished: all $done rounds done. Ask Claude to test the new AI." }
        elseif (Test-Path (Join-Path $camp 'stop-requested')) { Say "Stopped after round $done. Click Start to carry on." }
        else { Say "The trainer stopped unexpectedly after round $done. Ask Claude to look at the log in $camp." }
        Remove-Item (Join-Path $camp 'stop-requested') -ErrorAction SilentlyContinue
        Remove-Item $pidFile -ErrorAction SilentlyContinue
        break
    }
    if (GameRunning) {
        Say 'Slay the Spire 2 started: pausing training so the game runs smoothly.'
        New-Item (Join-Path $camp 'stop-requested') -ItemType File -Force | Out-Null
        & taskkill.exe /PID $p.Id /T /F | Out-Null
        continue
    }
    $now = RoundsDone
    if ($now -gt $seen) {
        $last = (Get-Content "$save.stats.jsonl" | Where-Object { $_.Trim() } | Select-Object -Last 1) | ConvertFrom-Json
        Say ("Round {0} of {1} saved (wins this round: {2:P0})." -f $now, $cfg.iterations, $last.win_rate)
        $seen = $now
    }
}
try { [Win.Power]::SetThreadExecutionState(0x80000000) | Out-Null } catch { }
Read-Host 'Press Enter to close'


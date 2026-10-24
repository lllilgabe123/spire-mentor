<#
  run-mp-test.ps1 -- local two-player multiplayer test for Spire Mentor.

  Starts two copies of Slay the Spire 2 on this PC with Steam switched off
  (so they use separate "default" test profiles, never the real Steam save):
    HOST   = "you": Spire Mentor + BaseLib + SpireTestBot
    CLIENT = "a friend without the mod":
             -Mode handshake : no mods at all (-nomods); checks it can join
             -Mode play      : only SpireTestBot (a non-gameplay mod, so the
                               game's mod check treats it like no mods), and
                               both bots play random moves for -Floors floors

  Test-only files go in <game>\mods_STEAMTEST (BaseLib + the bot) and are
  removed afterwards; <game>\mods is not touched. Results land in
  results\<timestamp>\ next to this script, ending with summary.txt.
#>
param(
    [ValidateSet('handshake', 'play')] [string]$Mode = 'play',
    [int]$Floors = 6,
    [int]$TimeoutMinutes = 30,
    # Extra Workshop item ids to load on the HOST only (e.g. 3810059379 = Host
    # Tools / "Spire Co-op Trainer"), to check they cause no mod mismatch.
    [string[]]$HostWorkshopIds = @(),
    # Play mode: the host uses Host Tools ("Spire Co-op Trainer") on itself
    # (gold, a card, a relic) and the script checks the friend's game agrees.
    [switch]$CoopTrainer,
    # With -CoopTrainer: the card to give (e.g. MIDNIGHT); random if empty.
    # The script then also checks the AI program never choked on that card.
    [string]$CoopTrainerCard = '',
    # With -CoopTrainer: give to the OTHER player instead of the host.
    [switch]$CoopTrainerOther,
    # The host bot plays Spire Mentor's recommended moves in fights (test
    # harness only) -- reaches elites/bosses, feeds the tracker.
    [switch]$FollowAi,
    # Run the test games and their AI program at below-normal CPU priority,
    # so whatever the user is playing meanwhile stays smooth.
    [switch]$LowPriority,
    [string]$GameDir = 'E:\Steam\steamapps\common\Slay the Spire 2'
)

$ErrorActionPreference = 'Stop'
if ($CoopTrainer -and $HostWorkshopIds -notcontains '3810059379') { $HostWorkshopIds += '3810059379' }
$here = $PSScriptRoot
$exe = Join-Path $GameDir 'SlayTheSpire2.exe'
$userDir = Join-Path $env:APPDATA 'SlayTheSpire2'
$testMods = Join-Path $GameDir 'mods_STEAMTEST'
$baseLibSrc = 'E:\Steam\steamapps\workshop\content\2868840\3737335127\BaseLib'
$botBuild = Join-Path $here 'bin\Release\net9.0\SpireTestBot.dll'

if (Get-Process SlayTheSpire2 -ErrorAction SilentlyContinue) {
    throw 'Slay the Spire 2 is running. Close it first: the test needs the game files and port 9002.'
}
if (-not (Test-Path $botBuild)) { throw "Bot not built: $botBuild" }

$stampTime = Get-Date
$stamp = $stampTime.ToString('yyyyMMdd-HHmmss')
$out = Join-Path $here "results\$stamp-$Mode"
New-Item -ItemType Directory -Force $out | Out-Null
$hostStatus = Join-Path $out 'host-status.jsonl'
$clientStatus = Join-Path $out 'client-status.jsonl'
$hostLog = Join-Path $out 'host-godot.log'
$clientLog = Join-Path $out 'client-godot.log'
Write-Host "Results: $out"

# --- test-only mods --------------------------------------------------------
if (Test-Path $testMods) { Remove-Item -Recurse -Force $testMods }
New-Item -ItemType Directory -Force (Join-Path $testMods 'SpireTestBot') | Out-Null
Copy-Item -Recurse $baseLibSrc (Join-Path $testMods 'BaseLib')
Copy-Item $botBuild, (Join-Path $here 'SpireTestBot.json') (Join-Path $testMods 'SpireTestBot')
$extraIds = @()
foreach ($wid in $HostWorkshopIds) {
    $src = "E:\Steam\steamapps\workshop\content\2868840\$wid"
    if (-not (Test-Path $src)) { throw "Workshop item $wid is not downloaded ($src)." }
    Copy-Item -Recurse $src (Join-Path $testMods "ws-$wid")
    $extraIds += @(Get-ChildItem $src -Recurse -Filter *.json | ForEach-Object {
            try { $j = Get-Content $_.FullName -Raw | ConvertFrom-Json } catch { return }
            if ($j.id -and ($j.PSObject.Properties.Name -contains 'has_dll')) { $j.id } })
}
# A crash must not leave test mods where the real game would load them.
trap {
    Get-Process SlayTheSpire2 -ErrorAction SilentlyContinue | Where-Object { $_.StartTime -ge $stampTime } | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep 2
    Remove-Item -Recurse -Force $testMods -ErrorAction SilentlyContinue
    break
}

# --- test profiles: windowed, muted, only the chosen mods --------------------
function Write-TestSettings([string]$accountId, [string[]]$enabled, [int]$x) {
    $real = Get-ChildItem (Join-Path $userDir 'steam') -Recurse -Filter settings.save |
        Where-Object { $_.FullName -notmatch 'couch' } | Select-Object -First 1
    $s = Get-Content $real.FullName -Raw | ConvertFrom-Json
    $s.fullscreen = $false
    $s.window_size = [pscustomobject]@{ X = 960; Y = 540 }
    $s.window_position = [pscustomobject]@{ X = $x; Y = 60 }
    $s.volume_master = 0
    $s.fps_limit = 60
    $s.limit_fps_in_background = $false
    $s.seen_ea_disclaimer = $true
    $s.skip_intro_logo = $true
    $mods = @(
        @{ id = 'BaseLib'; source = 'steam_workshop' },
        @{ id = 'SpireTestBot'; source = 'steam_workshop' },
        @{ id = 'SpireMentor'; source = 'mods_directory' },
        @{ id = 'DrawOrderOverlay'; source = 'mods_directory' }
    ) + @($extraIds | ForEach-Object { @{ id = $_; source = 'steam_workshop' } }) | ForEach-Object { [pscustomobject]@{ id = $_.id; is_enabled = ($enabled -contains $_.id); source = $_.source } }
    # mods_enabled = the one-time "you are running mods" notice was accepted;
    # without it the game skips every mod (ModManager: PlayerAgreedToModLoading).
    $s.mod_settings = [pscustomobject]@{ mods_enabled = $true; mod_list = @($mods) }
    $dir = Join-Path $userDir "default\$accountId"
    New-Item -ItemType Directory -Force $dir | Out-Null
    $path = Join-Path $dir 'settings.save'
    if ((Test-Path $path) -and -not (Test-Path "$path.before-mp-test")) { Copy-Item $path "$path.before-mp-test" }
    [IO.File]::WriteAllText($path, ($s | ConvertTo-Json -Depth 10), [Text.UTF8Encoding]::new($false))
}
Write-TestSettings '1' (@('BaseLib', 'SpireMentor', 'SpireTestBot') + $extraIds) 20
Write-TestSettings '1001' @('SpireTestBot') 1000

# --- launch -----------------------------------------------------------------
$hostArgs = @('--log-file', "`"$hostLog`"", '--force-steam', 'off', '--fastmp', 'host_standard',
    '--spiretestbot', '--spiretestbot-out', "`"$hostStatus`"", '--spiretestbot-floors', "$Floors", '--spiretestbot-probe')
if ($CoopTrainer) { $hostArgs += '--spiretestbot-cooptrainer' }
if ($CoopTrainer -and $CoopTrainerCard) { $hostArgs += @('--spiretestbot-cooptrainer-card', $CoopTrainerCard) }
if ($CoopTrainer -and $CoopTrainerOther) { $hostArgs += '--spiretestbot-cooptrainer-other' }
if ($FollowAi) { $hostArgs += '--spiretestbot-follow-ai' }
$advisorOut = Join-Path $env:LOCALAPPDATA 'SpireMentor\advisor_stdout.log'
$advisorOutStart = if (Test-Path $advisorOut) { (Get-Item $advisorOut).Length } else { 0 }
$clientArgs = @('--log-file', "`"$clientLog`"", '--force-steam', 'off', '--fastmp', 'join', '--clientId', '1001')
if ($Mode -eq 'handshake') { $clientArgs += '--nomods' }
else { $clientArgs += @('--spiretestbot', '--spiretestbot-out', "`"$clientStatus`"", '--spiretestbot-floors', "$Floors") }

$started = Get-Date
$hostP = Start-Process $exe -ArgumentList $hostArgs -WorkingDirectory $GameDir -PassThru
Write-Host "Host started (pid $($hostP.Id)); waiting for its lobby..."
$deadline = (Get-Date).AddSeconds(90)
while ((Get-Date) -lt $deadline -and -not ((Test-Path $hostStatus) -and (Select-String -Path $hostStatus -Pattern '"event":"lobby"' -Quiet))) {
    if ($hostP.HasExited) { break }
    Start-Sleep 2
}
$clientP = Start-Process $exe -ArgumentList $clientArgs -WorkingDirectory $GameDir -PassThru
Write-Host "Client started (pid $($clientP.Id))."

# --- wait -------------------------------------------------------------------
function Count([string]$path, [string]$pattern) {
    if (-not (Test-Path $path)) { return 0 }
    return @(Select-String -Path $path -Pattern $pattern).Count
}
function Has([string]$path, [string]$pattern) { return (Count $path $pattern) -gt 0 }
function Done-Events([string]$path) {
    if (-not (Test-Path $path)) { return $false }
    return [bool](Select-String -Path $path -Pattern '"event":"(done|game_over|left_run|error_popup|quit)"' -Quiet)
}
function Lower-Priority {
    if (-not $LowPriority) { return }
    Get-Process SlayTheSpire2, SimAdvisor -ErrorAction SilentlyContinue |
        Where-Object { $_.StartTime -ge $started -and $_.PriorityClass -ne 'BelowNormal' } |
        ForEach-Object { try { $_.PriorityClass = 'BelowNormal' } catch { } }
}
$deadline = $started.AddMinutes($TimeoutMinutes)
while ((Get-Date) -lt $deadline) {
    Lower-Priority
    if ($hostP.HasExited -and $clientP.HasExited) { break }
    if ($Mode -eq 'handshake') {
        $joined = (Test-Path $hostStatus) -and (Select-String -Path $hostStatus -Pattern '"players":2' -Quiet)
        $popup = (Has $hostLog 'Mismatch in gameplay-relevant mods|ModMismatch') -or (Has $clientLog 'Mismatch in gameplay-relevant mods|ModMismatch')
        if ($joined -or $popup) { Start-Sleep 10; break }
    }
    elseif ((Done-Events $hostStatus) -and (Done-Events $clientStatus)) { Start-Sleep 12; break }
    Start-Sleep 5
}
$timedOut = (Get-Date) -ge $deadline
foreach ($p in @($clientP, $hostP)) { if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
Start-Sleep 3
Get-Process SimAdvisor -ErrorAction SilentlyContinue | Where-Object { $_.StartTime -ge $started } | Stop-Process -Force -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $testMods -ErrorAction SilentlyContinue

# --- analyse ----------------------------------------------------------------
function Events([string]$path) {
    if (-not (Test-Path $path)) { return @() }
    return @(Get-Content $path | ForEach-Object { try { $_ | ConvertFrom-Json } catch { } })
}
$he = Events $hostStatus; $ce = Events $clientStatus
$lastFloor = { param($ev) ($ev | Where-Object { $_.event -eq 'map_vote' -or $_.event -eq 'done' } | ForEach-Object { $_.data.floor } | Measure-Object -Maximum).Maximum }

$checks = [ordered]@{}
$checks['Friend joined the host''s lobby (no mod mismatch)'] = [bool]($he | Where-Object { $_.event -eq 'lobby' -and $_.data.players -ge 2 })
$checks['No "mod mismatch" in either log'] = -not ((Has $hostLog 'Mismatch in gameplay-relevant mods|ModMismatch') -or (Has $clientLog 'Mismatch in gameplay-relevant mods|ModMismatch'))
$checks['Spire Mentor loaded on the host'] = Has $hostLog '=== SpireMentor ready ==='
foreach ($id in $extraIds) {
    $checks["Extra mod $id loaded on the host"] = Has $hostLog "Finished mod initialization for .*\($([regex]::Escape($id))\)"
}
if ($Mode -eq 'handshake') {
    $checks['Friend ran with no mods at all'] = Has $clientLog "'nomods' passed"
}
else {
    $checks['Run started on both, as multiplayer'] = [bool]($he | Where-Object { $_.event -eq 'run_started' -and $_.data.net -eq 'Host' }) -and [bool]($ce | Where-Object { $_.event -eq 'run_started' -and $_.data.net -eq 'Client' })
    $checks["Reached floor $Floors on both"] = [bool]($he | Where-Object event -eq 'done') -and [bool]($ce | Where-Object event -eq 'done')
    $checks['Games never fell out of sync'] = -not ((Has $hostLog 'State divergence') -or (Has $clientLog 'State divergence'))
    $checks['Spire Mentor switched to advice-only'] = Has $hostLog 'Multiplayer run: advice only'
    $checks['Spire Mentor saw decisions to advise on'] = (Count $hostLog '\[SimAdvisor\] Decision up') -gt 0
    $advLogs = @(Get-ChildItem (Join-Path $env:LOCALAPPDATA 'SpireMentor\logs') -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -ge $started })
    $advLines = ($advLogs | ForEach-Object { @(Get-Content $_.FullName -ErrorAction SilentlyContinue).Count } | Measure-Object -Sum).Sum
    $checks['AI program wrote advice records during the run'] = $advLines -gt 0
    $checks['AI moves refused (Ctrl+Shift+N)'] = Has $hostLog 'AI moves is off in multiplayer'
    $checks['AI auto-play refused (Ctrl+Shift+P)'] = Has $hostLog 'AI auto-play is off in multiplayer'
    $checks['Undo refused (Ctrl+Z)'] = Has $hostLog 'undo is off in multiplayer'
    $checks['Speed change refused (F10)'] = Has $hostLog 'Game speed is off in multiplayer'
    $checks['Bot hit no errors'] = -not [bool](($he + $ce) | Where-Object { $_.event -in 'error', 'error_popup' })

    if ($CoopTrainer) {
        $ct = $he | Where-Object event -eq 'coop_trainer' | Select-Object -First 1
        $checks['Co-op Trainer test ran on the host'] = [bool]$ct -and -not [bool]($he | Where-Object event -eq 'coop_trainer_error')
        if ($ct) {
            $r = @($ct.data.results)
            $who = if ($CoopTrainerOther) { 'Friend' } else { 'Host' }
            $checks["$who received +77 gold"] = [bool]($r | Where-Object { $_.what -like 'gold*' -and $_.gold_change -eq 77 })
            $checks["$who received the card"] = [bool]($r | Where-Object { $_.what -like 'card*' -and $_.deck_change -eq 1 })
            $checks["$who received the relic"] = [bool]($r | Where-Object { $_.what -like 'relic*' -and $_.relic_change -eq 1 })
            # Compare EVERY player as each game sees it, on every floor after
            # the gifts where both games took a snapshot.
            $hostSnaps = @{}; foreach ($s in ($he | Where-Object event -eq 'snapshot')) { $hostSnaps[[int]$s.data.floor] = $s }
            # Each game snapshots when ITS map opens, so a pick the other
            # player makes a second later can show as a one-floor difference.
            # Only a difference that is still there on the next compared floor
            # (or on the last one) is a real disagreement.
            $compared = 0; $mismatches = @(); $pending = @{}
            $clientSnaps = @($ce | Where-Object { $_.event -eq 'snapshot' -and [int]$_.data.floor -gt [int]$ct.data.floor -and $hostSnaps.ContainsKey([int]$_.data.floor) })
            for ($n = 0; $n -lt $clientSnaps.Count; $n++) {
                $s = $clientSnaps[$n]; $f = [int]$s.data.floor; $last = $n -eq $clientSnaps.Count - 1
                $compared++
                $now = @{}
                foreach ($seen in $s.data.players) {
                    $mine = $hostSnaps[$f].data.players | Where-Object id -eq $seen.id
                    foreach ($k in 'gold', 'deck', 'relics', 'relic_ids') {
                        if ("$($mine.$k)" -ne "$($seen.$k)") {
                            $key = "$($seen.id)/$k"
                            $now[$key] = "floor ${f} player $($seen.id) ${k}: host's game says $($mine.$k), friend's game says $($seen.$k)"
                            if ($pending.ContainsKey($key) -or $last) { $mismatches += $now[$key] }
                        }
                    }
                }
                $pending = $now
            }
            $checks["Both games agree on every player's gold/deck/relics ($compared floor(s) compared)"] = $compared -gt 0 -and $mismatches.Count -eq 0
            if (-not $CoopTrainerOther) {
                $checks["Friend's game logged the host obtaining the gifts"] = (Count $clientLog 'Player 1 obtained') -gt 0
            }
            if ($CoopTrainerCard) {
                $checks["Host received $CoopTrainerCard"] = [bool]($r | Where-Object { $_.what -eq "card $CoopTrainerCard" -and $_.deck_change -eq 1 })
                # Only this test's part of the AI program's log.
                $advText = ''
                if (Test-Path $advisorOut) {
                    $fs = [IO.File]::Open($advisorOut, 'Open', 'Read', 'ReadWrite')
                    try { $fs.Seek([Math]::Min($advisorOutStart, $fs.Length), 'Begin') | Out-Null; $advText = (New-Object IO.StreamReader($fs)).ReadToEnd() } finally { $fs.Close() }
                }
                $chokes = @([regex]::Matches($advText, "advisor error: [^\]]*$CoopTrainerCard")).Count
                $okAdvice = @([regex]::Matches($advText, 'advice #a\d+: best=(?!None)')).Count
                $checks["AI program never choked on $CoopTrainerCard ($chokes errors)"] = $chokes -eq 0
                $checks["AI program gave advice after the gift ($okAdvice pieces)"] = $okAdvice -gt 0
            }
        }
    }
}

$lines = @("Spire Mentor multiplayer test ($Mode) - $stamp", "Duration: $([int]((Get-Date) - $started).TotalMinutes) min$(if ($timedOut) { ' (TIMED OUT)' })", '')
foreach ($k in $checks.Keys) { $lines += ('{0}  {1}' -f $(if ($checks[$k]) { 'PASS' } else { 'FAIL' }), $k) }
$lines += ''
$lines += "Host floor reached: $(& $lastFloor $he)   Client floor reached: $(& $lastFloor $ce)"
$lines += "Spire Mentor errors/exceptions in host log: $(Count $hostLog '\[SpireMentor\].*(Exception|threw|failed)')"
foreach ($e in ($he | Where-Object { $_.event -in 'coop_trainer', 'coop_trainer_error' })) {
    $lines += "Co-op Trainer: $($e.data | ConvertTo-Json -Compress -Depth 5)"
}
if ($mismatches) { $lines += $mismatches | ForEach-Object { "  MISMATCH $_" } }
$lines += "Friend's log lines about the host obtaining things:"
$lines += @(if (Test-Path $clientLog) { Select-String -Path $clientLog -Pattern 'Player 1 obtained' | Select-Object -First 12 | ForEach-Object { '  ' + $_.Line.Trim() } })
foreach ($e in (($he + $ce) | Where-Object { $_.event -in 'error', 'error_popup', 'stuck', 'unknown_screen', 'modal' } | Select-Object -First 25)) {
    $lines += "  [$($e.role) $($e.t)] $($e.event): $(($e.data | ConvertTo-Json -Compress -Depth 4) -replace '\\n',' ' | ForEach-Object { if ($_.Length -gt 300) { $_.Substring(0, 300) + '...' } else { $_ } })"
}
$lines | Set-Content (Join-Path $out 'summary.txt') -Encoding utf8
$lines | ForEach-Object { Write-Host $_ }

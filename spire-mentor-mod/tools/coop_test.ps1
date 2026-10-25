# coop_test.ps1 -- a local co-op run against a dummy partner, on one machine.
#
# Starts two game instances with the game's own `--fastmp` dev launch
# (NMainMenu.CheckCommandLineArgs): ENet on 127.0.0.1:33771 instead of Steam
# lobbies, so no second Steam account is needed.
#
#   host   --fastmp host_standard        you play here (Spire Mentor as usual)
#   dummy  --fastmp join --clientId 1000 Spire Mentor plays it hands-free
#          --spire-mentor-autoplay       (SimCoopDummy readies the lobby, then
#                                        SimAdvisorAutoplay = F8 + F7)
#
# Pick your character in the host window and Embark; the run starts once the
# dummy is ready too. Plan and caveats: docs/plans/2026-10-23-coop-multiplayer.md.
#
# Both instances run under the same Steam account and so share one profile;
# the dummy writes no saves (--spire-mentor-no-saves, SimNoSaves) so the two
# never race on it.
#
# Both instances use the mod in <GameDir>/mods. The dummy's bridge port comes
# from --spire-mentor-port (both share SpireMentor.conf), and each instance
# logs to its own file (Godot's --log-file) instead of the shared godot.log.
#
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File spire-mentor-mod/tools/coop_test.ps1
#   ... -Character silent           the dummy's character (default: preselected)
#   ... -NoSteamDummy               start the dummy with --force-steam off; it
#                                   then sees no Workshop mods (BaseLib!)
#   ... -Speed 1                    game speed of both instances (default 3x)
#   ... -BotHost                    the host plays hands-free too (unattended
#                                   test of both sides; it embarks once the
#                                   dummy is in the lobby)

param(
    [string]$GameDir = $(if ($env:STS2_GAME_DIR) { $env:STS2_GAME_DIR } else { "C:/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2" }),
    [string]$Character = "",
    [int]$DummyPort = 9012,
    # Game speed for both instances (0.1-4.0; F9/F10 change it live).
    [double]$Speed = 3.0,
    [string]$LogDir = "",
    [switch]$NoSteamDummy,
    [switch]$BotHost
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $GameDir "SlayTheSpire2.exe"
if (-not (Test-Path $exe)) {
    Write-Error "Game not found: $exe (pass -GameDir or set STS2_GAME_DIR)."
}
if (-not $NoSteamDummy -and -not (Get-Process steam -ErrorAction SilentlyContinue)) {
    # A direct launch without a running Steam client fails Steamworks init
    # (ascension_climb.py launch_game, issue 9).
    Write-Error "Steam is not running; start it first."
}
if (Get-Process SlayTheSpire2 -ErrorAction SilentlyContinue) {
    Write-Error "A game instance is already running; close it first."
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
if (-not $LogDir) {
    $LogDir = Join-Path $repo ("logs/{0}-coop" -f (Get-Date -Format "yyyy-MM-dd-HHmmss"))
}
New-Item -ItemType Directory -Force $LogDir | Out-Null
$hostLog = Join-Path $LogDir "host.godot.log"
$dummyLog = Join-Path $LogDir "dummy.godot.log"

Write-Host "Starting host..."
$speedArg = $Speed.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$hostArgs = @("--fastmp", "host_standard", "--log-file", "`"$hostLog`"", "--spire-mentor-speed", $speedArg)
if ($BotHost) { $hostArgs += "--spire-mentor-autoplay" }
Start-Process -FilePath $exe -WorkingDirectory $GameDir -ArgumentList $hostArgs | Out-Null

# The dummy may only join once the host's ENet socket is up.
$deadline = (Get-Date).AddSeconds(120)
while (-not (Get-NetUDPEndpoint -LocalPort 33771 -ErrorAction SilentlyContinue)) {
    if ((Get-Date) -gt $deadline) {
        Write-Error "The host never opened its co-op lobby (UDP 33771). See $hostLog"
    }
    Start-Sleep -Seconds 1
}

Write-Host "Host lobby is up; starting the dummy..."
$dummyArgs = @(
    "--fastmp", "join", "--clientId", "1000",
    "--spire-mentor-autoplay",
    "--spire-mentor-no-saves",
    "--spire-mentor-port", "$DummyPort",
    "--spire-mentor-speed", $speedArg,
    "--log-file", "`"$dummyLog`""
)
if ($Character) { $dummyArgs += @("--spire-mentor-character", $Character) }
if ($NoSteamDummy) { $dummyArgs += @("--force-steam", "off") }
Start-Process -FilePath $exe -WorkingDirectory $GameDir -ArgumentList $dummyArgs | Out-Null

Write-Host ""
if (-not $BotHost) { Write-Host "Pick a character in the HOST window and Embark." }
Write-Host "Logs:"
Write-Host "  host   $hostLog"
Write-Host "  dummy  $dummyLog"
Write-Host "Mod lines only:  Select-String '\[SpireMentor\]' '$dummyLog' | Select-Object -Last 40"

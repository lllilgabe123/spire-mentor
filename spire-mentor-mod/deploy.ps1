# deploy.ps1 -- Standalone deploy of SpireMentor into the game's mods folder,
# without a rebuild. `dotnet build` already runs this via the DeploySpireMentor
# MSBuild target (SpireMentor.csproj); use this script when you just want to
# re-copy the manifest/config template after editing them, or to deploy to a
# non-default game install.
#
# Usage:
#   ./deploy.ps1
#   ./deploy.ps1 -GameDir "D:/SteamLibrary/steamapps/common/Slay the Spire 2"

param(
    # Default: the game dir SpireMentor.csproj discovers (STS2_GAME_DIR, else
    # every Steam library), so the build and this script never disagree.
    [string]$GameDir = ""
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $GameDir) {
    $GameDir = (dotnet msbuild (Join-Path $scriptDir "SpireMentor.csproj") -getProperty:STS2_GAME_DIR).Trim()
}
$modsDir = Join-Path $GameDir "mods/SpireMentor"

if (-not (Test-Path $GameDir)) {
    Write-Error "Game directory not found: $GameDir. Pass -GameDir to override."
}

$dll = Join-Path $scriptDir "bin/Debug/net9.0/SpireMentor.dll"
if (-not (Test-Path $dll)) {
    # Fall back to Release if that's what was last built.
    $dll = Join-Path $scriptDir "bin/Release/net9.0/SpireMentor.dll"
}
if (-not (Test-Path $dll)) {
    Write-Error "SpireMentor.dll not found under bin/. Run 'dotnet build' first."
}

New-Item -ItemType Directory -Force -Path $modsDir | Out-Null
Copy-Item -Path $dll -Destination $modsDir -Force
Copy-Item -Path (Join-Path $scriptDir "SpireMentor.json") -Destination $modsDir -Force
Copy-Item -Path (Join-Path $scriptDir "SpireMentor.conf.template") -Destination $modsDir -Force

Write-Host "Deployed SpireMentor to $modsDir"
Get-ChildItem $modsDir

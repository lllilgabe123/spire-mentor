# Assemble the distributable SpireMentor mod zip (GH #380).
#
# Output: SpireMentor-v<version>-win64.zip whose root contains mods/, so the
# install instruction is exactly "extract into the Slay the Spire 2 folder".
#
#   mods/SpireMentor/   SpireMentor.dll, SpireMentor.json, SpireMentor.conf.template,
#                     SimAdvisor/            (the packaged torch-free advisor)
#   mods/BaseLib/     BaseLib.dll, BaseLib.pck, BaseLib.json,
#                     LICENSE-BaseLib.txt    (MIT notice; only staged when the
#                                             upstream license text is present
#                                             locally -- see below)
#   README.txt
#
# BaseLib is staged from the LOCAL GAME INSTALL: that is the exact v3.4.1
# binary this SpireMentor.dll was compiled against (the csproj references the
# deployed copy). BaseLib is MIT-licensed, so redistribution needs its
# license text in the zip. That text is NOT vendored in this repo (the
# authoritative copyright line lives upstream); it is picked up from an
# optional local clone of github.com/Alchyr/BaseLib-StS2 at refs/BaseLib-StS2
# (gitignored). Without the clone the zip builds with a WARNING and no
# LICENSE-BaseLib.txt -- fine for local testing, but source the license text
# from the BaseLib-StS2 upstream before distributing the zip.
#
# The SimAdvisor exe is taken prebuilt from sts2-pycli/dist/SimAdvisor by
# default; pass -BuildAdvisor to rebuild it first (slow-ish).
#
# NOTE (licensing): SpireMentor is MIT (spire-mentor-mod/LICENSE); the zip also
# bundles BaseLib (MIT, its notice below) and the advisor exe.
param(
    # Rebuild the advisor exe before packaging instead of reusing the
    # prebuilt sts2-pycli/dist/SimAdvisor.
    [switch]$BuildAdvisor
)
$ErrorActionPreference = "Stop"
$bridge = $PSScriptRoot
$repo = Split-Path -Parent $bridge
$pycli = Join-Path $repo "sts2-pycli"

# --- locate the game (asks SpireMentor.csproj: STS2_GAME_DIR, else every Steam library)
$gameDir = (dotnet msbuild (Join-Path $bridge "SpireMentor.csproj") -getProperty:STS2_GAME_DIR).Trim()
$baseLibDir = Join-Path $gameDir "mods/BaseLib"
if (-not (Test-Path (Join-Path $baseLibDir "BaseLib.dll"))) {
    throw "BaseLib not found at $baseLibDir -- install BaseLib into the game first " +
          "(https://github.com/Alchyr/BaseLib-StS2/releases, v3.4.1 for game v0.107.1), " +
          "or set STS2_GAME_DIR."
}
# The MIT license text is not vendored here -- it must come from the upstream
# BaseLib-StS2 repo (optional gitignored clone under refs/). Missing text is a
# warning, not an error: the zip still builds, but must not be distributed
# until LICENSE-BaseLib.txt is sourced from upstream.
$baseLibLicense = Join-Path $repo "refs/BaseLib-StS2/LICENSE.txt"
$haveBaseLibLicense = Test-Path $baseLibLicense
if (-not $haveBaseLibLicense) {
    Write-Warning ("BaseLib license text not found at $baseLibLicense -- the zip will NOT " +
        "contain LICENSE-BaseLib.txt. Before distributing, clone " +
        "https://github.com/Alchyr/BaseLib-StS2 into refs/ and re-run (the MIT notice must ship).")
}

# --- build both halves -------------------------------------------------------
dotnet build $bridge -c Release -p:SpireMentorSkipDeploy=true
if ($LASTEXITCODE -ne 0) { throw "dotnet build failed" }
$dll = Join-Path $bridge "bin/Release/net9.0/SpireMentor.dll"
if (-not (Test-Path $dll)) { throw "missing $dll" }

$advisorDist = Join-Path $pycli "dist/SimAdvisor"
if ($BuildAdvisor) {
    pwsh -NoProfile -File (Join-Path $pycli "scripts/build_advisor_exe.ps1")
    if ($LASTEXITCODE -ne 0) { throw "advisor exe build failed" }
}
if (-not (Test-Path (Join-Path $advisorDist "SimAdvisor.exe"))) {
    throw "missing $advisorDist\SimAdvisor.exe -- no prebuilt advisor found; build one with " +
          "-BuildAdvisor (runs sts2-pycli/scripts/build_advisor_exe.ps1)"
}

# --- stage -------------------------------------------------------------------
$version = (Get-Content (Join-Path $bridge "SpireMentor.json") | ConvertFrom-Json).version
$staging = Join-Path $bridge "staging"
if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
$simDir = New-Item -ItemType Directory -Path (Join-Path $staging "mods/SpireMentor") -Force
$libDir = New-Item -ItemType Directory -Path (Join-Path $staging "mods/BaseLib") -Force

Copy-Item $dll $simDir
Copy-Item (Join-Path $bridge "SpireMentor.json") $simDir
Copy-Item (Join-Path $bridge "SpireMentor.conf.template") $simDir
Copy-Item -Recurse $advisorDist (Join-Path $simDir "SimAdvisor")
Copy-Item (Join-Path $baseLibDir "BaseLib.dll") $libDir
Copy-Item (Join-Path $baseLibDir "BaseLib.pck") $libDir
Copy-Item (Join-Path $baseLibDir "BaseLib.json") $libDir
if ($haveBaseLibLicense) {
    Copy-Item $baseLibLicense (Join-Path $libDir "LICENSE-BaseLib.txt")
}
Copy-Item (Join-Path $bridge "packaging/README-dist.txt") (Join-Path $staging "README.txt")

# --- zip ---------------------------------------------------------------------
$zip = Join-Path $bridge "SpireMentor-v$version-win64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $zip
$sizeMB = [math]::Round((Get-Item $zip).Length / 1MB)
Write-Host "OK: $zip ($sizeMB MB)"
if (-not $haveBaseLibLicense) {
    Write-Warning "zip lacks LICENSE-BaseLib.txt -- do not distribute until the MIT notice is added."
}

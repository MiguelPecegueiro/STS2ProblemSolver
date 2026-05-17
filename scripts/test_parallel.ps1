# Validate two STS2 headless processes with separate STS2MCP ports (same Steam install).
#
# STS2 scans every subdirectory under mods/ for mod manifests. A folder renamed to
# *_disabled is still loaded if it contains STS2_MCP.json with id STS2_MCP - that causes
# "mod is already loaded with that name" even when only one folder looks "active".
#
# Fix: keep the second mod copy outside mods/ (_sts2_parallel_staging) and move it in
# only when launching that instance. Wait for MCP before swapping folders for the next.
#
# Usage: .\scripts\test_parallel.ps1 [-GameRoot "D:\SteamLibrary\steamapps\common\Slay the Spire 2"]

param(
    [string]$GameRoot = "D:\SteamLibrary\steamapps\common\Slay the Spire 2",
    [int]$Port0 = 15526,
    [int]$Port1 = 15527,
    [int]$BootTimeoutSeconds = 180,
    [int]$PollIntervalSeconds = 3,
    [int]$WaitSeconds = 30,
    [switch]$KeepProcesses
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "[test_parallel] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "[test_parallel] OK: $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[test_parallel] WARN: $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "[test_parallel] FAIL: $msg" -ForegroundColor Red }

$Exe = Join-Path $GameRoot "SlayTheSpire2.exe"
$ModsRoot = Join-Path $GameRoot "mods"
$Mod0 = Join-Path $ModsRoot "sts2mcp"
$Mod1 = Join-Path $ModsRoot "sts2mcp_1"
$StagingRoot = Join-Path $GameRoot "_sts2_parallel_staging"
$StagingMod0 = Join-Path $StagingRoot "sts2mcp"
$StagingMod1 = Join-Path $StagingRoot "sts2mcp_1"

$proc0 = $null
$proc1 = $null
$exitCode = 1

function Stop-Sts2Processes {
    Get-Process -Name "SlayTheSpire2" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

function Remove-LegacyDisabledModFolders {
    foreach ($name in @("_sts2mcp_disabled", "_sts2mcp_1_disabled")) {
        $path = Join-Path $ModsRoot $name
        if (Test-Path $path) {
            Write-Step "Removing legacy mods\$name (still scanned by the game)..."
            Remove-Item $path -Recurse -Force
        }
    }
}

function Get-McpModFoldersInMods {
    if (-not (Test-Path $ModsRoot)) { return @() }
    Get-ChildItem $ModsRoot -Directory -ErrorAction SilentlyContinue | Where-Object {
        Test-Path (Join-Path $_.FullName "STS2_MCP.json")
    }
}

function Assert-OnlyMcpModInMods {
    param([string]$ExpectedFolderName)
    $found = @(Get-McpModFoldersInMods)
    if ($found.Count -eq 0) {
        throw "No STS2_MCP mod under mods\ (expected mods\$ExpectedFolderName)."
    }
    if ($found.Count -gt 1) {
        $names = ($found | ForEach-Object { $_.Name }) -join ", "
        throw "Multiple STS2_MCP mods under mods\: $names. Only one folder may exist there at launch."
    }
    if ($found[0].Name -ne $ExpectedFolderName) {
        throw "Expected only mods\$ExpectedFolderName but found mods\$($found[0].Name)."
    }
}

function Move-ModFolder {
    param([string]$From, [string]$To)
    if (-not (Test-Path $From)) { return }
    if (Test-Path $To) { Remove-Item $To -Recurse -Force }
    $parent = Split-Path $To -Parent
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path $From -Destination $To
}

function Park-Mod1 {
    if (Test-Path $Mod1) { Move-ModFolder -From $Mod1 -To $StagingMod1 }
}

function Park-Mod0 {
    if (Test-Path $Mod0) { Move-ModFolder -From $Mod0 -To $StagingMod0 }
}

function Install-Mod0 {
    if ((Test-Path $StagingMod0) -and -not (Test-Path $Mod0)) {
        Move-ModFolder -From $StagingMod0 -To $Mod0
    }
}

function Install-Mod1 {
    if ((Test-Path $StagingMod1) -and -not (Test-Path $Mod1)) {
        Move-ModFolder -From $StagingMod1 -To $Mod1
    }
}

function Restore-AllModFolders {
    Install-Mod0
    Install-Mod1
    Park-Mod1
}

function Test-McpPort {
    param([int]$Port)
    $url = "http://127.0.0.1:$Port/"
    try {
        $body = $null
        if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
            $body = (curl.exe -s -m 5 $url 2>$null)
        }
        if (-not $body) {
            $resp = Invoke-WebRequest -Uri $url -TimeoutSec 5 -UseBasicParsing
            $body = $resp.Content
        }
        $json = $body | ConvertFrom-Json
        if ($json.status -eq "ok") {
            return @{ Ok = $true; Message = $json.message; Url = $url; Body = $body }
        }
        return @{ Ok = $false; Message = "Unexpected JSON: $body"; Url = $url; Body = $body }
    }
    catch {
        return @{ Ok = $false; Message = $_.Exception.Message; Url = $url; Body = $null }
    }
}

function Wait-ForMcpPort {
    param(
        [int]$Port,
        [System.Diagnostics.Process]$Process,
        [string]$Label,
        [int]$TimeoutSeconds = $BootTimeoutSeconds
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $attempt = 0
    while ((Get-Date) -lt $deadline) {
        if ($Process.HasExited) {
            Write-Fail "$Label process exited before MCP on port $Port (exit $($Process.ExitCode))"
            return $false
        }
        $attempt++
        $r = Test-McpPort -Port $Port
        if ($r.Ok) {
            Write-Ok "$Label MCP ready on port $Port after $attempt probe(s) - $($r.Message)"
            return $true
        }
        if ($attempt -eq 1 -or ($attempt % 10) -eq 0) {
            Write-Step "$Label waiting for MCP on port $Port... ($attempt)"
        }
        Start-Sleep -Seconds $PollIntervalSeconds
    }
    Write-Warn "$Label MCP on port $Port did not respond within ${TimeoutSeconds}s"
    return $false
}

function Invoke-ParallelTest {
    if (-not (Test-Path $Exe)) {
        throw "Game executable not found: $Exe`nSet -GameRoot to your STS2 install path."
    }
    if (-not (Test-Path $Mod0)) {
        throw "Expected mod folder not found: $Mod0"
    }
    if (-not (Get-Process -Name "steam" -ErrorAction SilentlyContinue)) {
        Write-Warn "Steam does not appear to be running - STS2 may not initialize Steamworks/MCP."
    }

    Write-Step "Stopping any existing SlayTheSpire2 processes..."
    Stop-Sts2Processes
    Start-Sleep -Seconds 2

    Remove-LegacyDisabledModFolders
    Restore-AllModFolders

    Write-Step "Preparing staged copy (port $Port1) outside mods\..."
    if (Test-Path $StagingRoot) { Remove-Item $StagingRoot -Recurse -Force }
    New-Item -ItemType Directory -Path $StagingRoot -Force | Out-Null
    Copy-Item $Mod0 $StagingMod1 -Recurse
    @{ port = $Port1 } | ConvertTo-Json | Set-Content -Path (Join-Path $StagingMod1 "STS2_MCP.conf") -Encoding UTF8
    @{ port = $Port0 } | ConvertTo-Json | Set-Content -Path (Join-Path $Mod0 "STS2_MCP.conf") -Encoding UTF8
    Park-Mod1
    Write-Ok "Second mod at $StagingMod1 (not under mods\)"

    Write-Step "Launching instance 0 (only mods\sts2mcp, port $Port0)..."
    Assert-OnlyMcpModInMods -ExpectedFolderName "sts2mcp"
    $script:proc0 = Start-Process -FilePath $Exe -ArgumentList @("--headless", "--quiet") -PassThru -WindowStyle Hidden
    Write-Ok "Instance 0 PID $($proc0.Id) started"

    $ready0 = Wait-ForMcpPort -Port $Port0 -Process $proc0 -Label "Instance 0"
    if ($ready0) {
        Write-Ok "Instance 0 finished mod load (MCP on $Port0)"
    } else {
        Write-Warn "Instance 0 MCP not ready - check game log for mod errors"
    }

    Write-Step "Launching instance 1 (park sts2mcp, install sts2mcp_1, port $Port1)..."
    Park-Mod0
    Install-Mod1
    Assert-OnlyMcpModInMods -ExpectedFolderName "sts2mcp_1"
    $script:proc1 = Start-Process -FilePath $Exe -ArgumentList @("--headless", "--quiet") -PassThru -WindowStyle Hidden
    Write-Ok "Instance 1 PID $($proc1.Id) started"

    $ready1 = Wait-ForMcpPort -Port $Port1 -Process $proc1 -Label "Instance 1"
    if ($ready1) {
        Write-Ok "Instance 1 finished mod load (MCP on $Port1)"
    } else {
        Write-Warn "Instance 1 MCP not ready - check game log for mod errors"
    }

    Install-Mod0
    Write-Ok "Restored mods\sts2mcp (instance 0 still uses its in-memory mod on $Port0)"

    Write-Step "Waiting $WaitSeconds seconds before final probe..."
    Start-Sleep -Seconds $WaitSeconds

    Write-Step "Probing MCP endpoints..."
    $r0 = Test-McpPort -Port $Port0
    $r1 = Test-McpPort -Port $Port1

    Write-Host ""
    Write-Host "=== Results ===" -ForegroundColor White
    @(
        @{ Label = "Instance 0 (port $Port0)"; Result = $r0 },
        @{ Label = "Instance 1 (port $Port1)"; Result = $r1 }
    ) | ForEach-Object {
        $r = $_.Result
        if ($r.Ok) {
            Write-Ok "$($_.Label) - $($r.Url) - $($r.Message)"
        } else {
            Write-Fail "$($_.Label) - $($r.Url) - $($r.Message)"
        }
    }

    $alive0 = -not $proc0.HasExited
    $alive1 = -not $proc1.HasExited
    Write-Host ""
    Write-Host "Processes: instance0 PID $($proc0.Id) alive=$alive0 | instance1 PID $($proc1.Id) alive=$alive1"

    $bothOk = $r0.Ok -and $r1.Ok -and $alive0 -and $alive1
    Write-Host ""
    if ($bothOk) {
        Write-Ok "Both MCP ports responded - parallel headless + separate ports looks viable."
        if ($KeepProcesses) {
            Write-Step "KeepProcesses set - leaving game processes running."
            $script:proc0 = $null
            $script:proc1 = $null
        }
        return 0
    }

    Write-Fail "Not all checks passed (HTTP and/or process exit)."
    if (-not $ready0 -or -not $ready1) {
        Write-Warn "If MCP never came up: launch STS2 once normally, accept the mod, then retry headless."
    }
    return 1
}

try {
    $exitCode = Invoke-ParallelTest
}
catch {
    Write-Fail $_.Exception.Message
    $exitCode = 1
}

Write-Step "Cleanup..."
Stop-Sts2Processes
Restore-AllModFolders
Remove-LegacyDisabledModFolders
if (-not $KeepProcesses) {
    if ($proc0 -and -not $proc0.HasExited) { Stop-Process -Id $proc0.Id -Force -ErrorAction SilentlyContinue }
    if ($proc1 -and -not $proc1.HasExited) { Stop-Process -Id $proc1.Id -Force -ErrorAction SilentlyContinue }
    Stop-Sts2Processes
}

exit $exitCode

# Launch N headless STS2 instances + agents for parallel training data collection.
#
# Mod staging matches test_parallel.ps1: only one STS2_MCP folder under mods\ at each
# game launch. Staged copies live in _sts2_parallel_staging\ outside mods\.
#
# Usage:
#   .\scripts\launch_parallel.ps1
#   .\scripts\launch_parallel.ps1 -Count 4 -Policy -WaitSeconds 60
#   (-Policy adds --policy, --card-reward-model, --ppo-model to each agent)
#
# All instances share one Steam profile save folder. Agents use --instance-id which
# enables abandoning a stale continue/abandon menu to reach singleplayer (recovery
# after crashes). Prefer -Count 4; more instances contend for the same save slot.
# Ctrl+C stops all processes and runs: py tools\merge_training_shards.py

param(
    [int]$Count = 4,
    [string]$GameRoot = "D:\SteamLibrary\steamapps\common\Slay the Spire 2",
    [int]$BasePort = 15526,
    [int]$BootTimeoutSeconds = 180,
    [int]$PollIntervalSeconds = 3,
    [int]$WaitSeconds = 60,
    [switch]$Policy
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "[launch_parallel] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "[launch_parallel] OK: $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[launch_parallel] WARN: $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "[launch_parallel] FAIL: $msg" -ForegroundColor Red }

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Exe = Join-Path $GameRoot "SlayTheSpire2.exe"
$ModsRoot = Join-Path $GameRoot "mods"
$StagingRoot = Join-Path $GameRoot "_sts2_parallel_staging"
$MergeScript = Join-Path $RepoRoot "tools\merge_training_shards.py"

$GameProcs = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()
$AgentProcs = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()

function Get-ModFolderName([int]$Index) {
    if ($Index -eq 0) { "sts2mcp" } else { "sts2mcp_$Index" }
}

function Get-ModPath([int]$Index) {
    Join-Path $ModsRoot (Get-ModFolderName $Index)
}

function Get-StagingModPath([int]$Index) {
    Join-Path $StagingRoot (Get-ModFolderName $Index)
}

function Stop-AllGameProcesses {
    Get-Process -Name "SlayTheSpire2" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

function Stop-TrackedAgents {
    foreach ($proc in $AgentProcs) {
        if ($proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
    $AgentProcs.Clear()
}

function Remove-LegacyDisabledModFolders {
    foreach ($name in @("_sts2mcp_disabled", "_sts2mcp_1_disabled")) {
        $path = Join-Path $ModsRoot $name
        if (Test-Path $path) {
            Write-Step "Removing legacy mods\$name ..."
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
        throw "Multiple STS2_MCP mods under mods\: $names"
    }
    if ($found[0].Name -ne $ExpectedFolderName) {
        throw "Expected mods\$ExpectedFolderName but found mods\$($found[0].Name)."
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

function Park-ModInMods {
    param([int]$Index)
    $modsPath = Get-ModPath $Index
    $stagingPath = Get-StagingModPath $Index
    if (Test-Path $modsPath) {
        Move-ModFolder -From $modsPath -To $stagingPath
    }
}

function Install-ModToMods {
    param([int]$Index)
    $modsPath = Get-ModPath $Index
    $stagingPath = Get-StagingModPath $Index
    if ((Test-Path $stagingPath) -and -not (Test-Path $modsPath)) {
        Move-ModFolder -From $stagingPath -To $modsPath
    }
}

function Park-AllModsFromMods {
    for ($i = 0; $i -lt $Count; $i++) {
        Park-ModInMods $i
    }
}

function Restore-CanonicalModLayout {
    Install-ModToMods 0
    for ($i = 1; $i -lt $Count; $i++) {
        Park-ModInMods $i
    }
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
            return $true
        }
    }
    catch { }
    return $false
}

function Wait-ForMcpPort {
    param(
        [int]$Port,
        [System.Diagnostics.Process]$Process,
        [string]$Label
    )
    $deadline = (Get-Date).AddSeconds($BootTimeoutSeconds)
    $attempt = 0
    while ((Get-Date) -lt $deadline) {
        if ($Process.HasExited) {
            throw "$Label process exited before MCP on port $Port (exit $($Process.ExitCode))"
        }
        $attempt++
        if (Test-McpPort -Port $Port) {
            Write-Ok "$Label MCP ready on port $Port (probe $attempt)"
            return
        }
        if ($attempt -eq 1 -or ($attempt % 10) -eq 0) {
            Write-Step "$Label waiting for MCP on port $Port... ($attempt)"
        }
        Start-Sleep -Seconds $PollIntervalSeconds
    }
    throw "$Label MCP on port $Port did not respond within ${BootTimeoutSeconds}s"
}

function Start-Agent {
    param([int]$Index, [int]$Port)
    $dataDir = "data/instances/$Index"
    $dataPath = Join-Path $RepoRoot $dataDir
    if (-not (Test-Path $dataPath)) {
        New-Item -ItemType Directory -Path $dataPath -Force | Out-Null
    }

    $agentArgs = @(
        "-m", "sts2_agent.main",
        "--port", $Port,
        "--instance-id", $Index,
        "--data-dir", $dataDir,
        "--no-compendium"
    )
    if ($Policy) {
        $agentArgs += @(
            "--policy",
            "--card-reward-model", $CardRewardModel,
            "--ppo-model", $PpoModel
        )
    }

    $proc = Start-Process -FilePath "py" -ArgumentList $agentArgs -PassThru -WorkingDirectory $RepoRoot
    $AgentProcs.Add($proc) | Out-Null
    Write-Ok "Agent instance $Index PID $($proc.Id) (port $Port, $dataDir)"
}

function Invoke-Merge {
    if (-not (Test-Path $MergeScript)) {
        Write-Warn "Merge script not found: $MergeScript"
        return
    }
    Write-Step "Merging training shards..."
    & py $MergeScript
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "merge_training_shards.py exited with code $LASTEXITCODE"
    }
}

function Invoke-Cleanup {
    Write-Step "Stopping agents..."
    Stop-TrackedAgents
    Write-Step "Stopping games..."
    foreach ($proc in $GameProcs) {
        if ($proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
    $GameProcs.Clear()
    Stop-AllGameProcesses
    Write-Step "Restoring mod folders..."
    Remove-LegacyDisabledModFolders
    if (Test-Path (Get-ModPath 0)) {
        Restore-CanonicalModLayout
    } else {
        Install-ModToMods 0
    }
}

if ($Count -lt 1) {
    throw "-Count must be at least 1"
}
if ($Count -gt 4) {
    Write-Warn "-Count $Count > 4: extra instances often see continue/abandon_run menu (shared Steam save). Prefer -Count 4."
}

try {
    if (-not (Test-Path $Exe)) {
        throw "Game executable not found: $Exe"
    }
    if (-not (Test-Path (Get-ModPath 0))) {
        throw "Expected mod folder not found: $(Get-ModPath 0)"
    }
    if (-not (Get-Process -Name "steam" -ErrorAction SilentlyContinue)) {
        Write-Warn "Steam does not appear to be running."
    }

    Write-Step "Stopping existing SlayTheSpire2 processes..."
    Stop-AllGameProcesses
    Start-Sleep -Seconds 2
    Remove-LegacyDisabledModFolders
    Park-AllModsFromMods
    Install-ModToMods 0

    Write-Step "Preparing $Count staged mod(s) in $StagingRoot ..."
    if (Test-Path $StagingRoot) { Remove-Item $StagingRoot -Recurse -Force }
    New-Item -ItemType Directory -Path $StagingRoot -Force | Out-Null

    $canonical = Get-ModPath 0
    @{ port = $BasePort } | ConvertTo-Json | Set-Content -Path (Join-Path $canonical "STS2_MCP.conf") -Encoding UTF8

    for ($i = 1; $i -lt $Count; $i++) {
        $port = $BasePort + $i
        $staging = Get-StagingModPath $i
        Copy-Item $canonical $staging -Recurse
        @{ port = $port } | ConvertTo-Json | Set-Content -Path (Join-Path $staging "STS2_MCP.conf") -Encoding UTF8
        Write-Ok "Staged $(Get-ModFolderName $i) port $port"
    }

    for ($i = 1; $i -lt $Count; $i++) {
        Park-ModInMods $i
    }
    Assert-OnlyMcpModInMods -ExpectedFolderName (Get-ModFolderName 0)

    for ($i = 0; $i -lt $Count; $i++) {
        $port = $BasePort + $i
        $folderName = Get-ModFolderName $i

        if ($i -gt 0) {
            for ($j = 0; $j -lt $Count; $j++) {
                Park-ModInMods $j
            }
            Install-ModToMods $i
            Assert-OnlyMcpModInMods -ExpectedFolderName $folderName
        }

        Write-Step "Launching game instance $i (mods\$folderName, port $port)..."
        $gameProc = Start-Process -FilePath $Exe -ArgumentList @("--headless", "--quiet") -PassThru -WindowStyle Hidden
        $GameProcs.Add($gameProc) | Out-Null
        Write-Ok "Game instance $i PID $($gameProc.Id)"

        Wait-ForMcpPort -Port $port -Process $gameProc -Label "Game $i"
        Start-Agent -Index $i -Port $port
    }

    Install-ModToMods 0
    if ($WaitSeconds -gt 0) {
        Write-Step "Soak $WaitSeconds seconds before collection monitor..."
        Start-Sleep -Seconds $WaitSeconds
    }
    Write-Ok "All $Count instance(s) running. Press Ctrl+C to stop and merge shards."

    while ($true) {
        Start-Sleep -Seconds 5
        foreach ($proc in $GameProcs) {
            if ($proc.HasExited) {
                Write-Warn "Game PID $($proc.Id) exited (code $($proc.ExitCode))"
            }
        }
        foreach ($proc in $AgentProcs) {
            if ($proc.HasExited) {
                Write-Warn "Agent PID $($proc.Id) exited (code $($proc.ExitCode))"
            }
        }
    }
}
catch [System.OperationCanceledException] {
    Write-Step "Interrupted."
}
catch {
    Write-Fail $_.Exception.Message
}
finally {
    Invoke-Cleanup
    Invoke-Merge
}

<#
.SYNOPSIS
Collects Windows and Intel AI PC hardware information as JSON.

.EXAMPLE
.\tools\performance\collect_intel_environment.ps1 -OutputPath .\environment.json
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-UniqueTextValues {
    param([object[]]$Values)
    return @(
        $Values |
            ForEach-Object { if ($_ -ne $null) { ([string]$_).Trim() } } |
            Where-Object { $_ } |
            Sort-Object -Unique
    )
}

function Get-CimValuesSafely {
    param([string]$ClassName)
    try {
        return @(Get-CimInstance -ClassName $ClassName -ErrorAction Stop)
    } catch {
        return @()
    }
}

$processors = @(Get-CimValuesSafely -ClassName "Win32_Processor")
$computers = @(Get-CimValuesSafely -ClassName "Win32_ComputerSystem")
$operatingSystems = @(Get-CimValuesSafely -ClassName "Win32_OperatingSystem")
$videoControllers = @(Get-CimValuesSafely -ClassName "Win32_VideoController")
$computer = if ($computers.Count -gt 0) { $computers[0] } else { $null }
$operatingSystem = if ($operatingSystems.Count -gt 0) { $operatingSystems[0] } else { $null }

$npuNames = @()
try {
    $npuNames = @(
        Get-PnpDevice -PresentOnly -ErrorAction Stop |
            Where-Object {
                $_.FriendlyName -match '(?i)\bNPU\b|Neural Processing|AI Boost' -or
                $_.InstanceId -match '(?i)(^|[\\&#_-])(NPU|VPU)([\\&#_-]|$)'
            } |
            ForEach-Object { $_.FriendlyName }
    )
} catch {
    $npuNames = @()
}

$powerPlan = $null
try {
    $powerPlan = ((& powercfg /getactivescheme 2>$null) | Out-String).Trim()
    if (-not $powerPlan) {
        $powerPlan = $null
    }
} catch {
    $powerPlan = $null
}

$cpuNames = @(Get-UniqueTextValues -Values @($processors | ForEach-Object { $_.Name }))
if ($cpuNames.Count -eq 0 -and $env:PROCESSOR_IDENTIFIER) {
    $cpuNames = @($env:PROCESSOR_IDENTIFIER)
}
$gpuNames = @(Get-UniqueTextValues -Values @($videoControllers | ForEach-Object { $_.Name }))
$npuNames = @(Get-UniqueTextValues -Values $npuNames)

$physicalCores = $null
$logicalThreads = $null
if ($processors.Count -gt 0) {
    $physicalCores = [int](($processors | Measure-Object NumberOfCores -Sum).Sum)
    $logicalThreads = [int](($processors | Measure-Object NumberOfLogicalProcessors -Sum).Sum)
}

$totalMemoryBytes = $null
$totalMemoryGib = $null
if ($computer -ne $null -and $computer.TotalPhysicalMemory -ne $null) {
    $totalMemoryBytes = [int64]$computer.TotalPhysicalMemory
    $totalMemoryGib = [math]::Round(([double]$computer.TotalPhysicalMemory / 1GB), 2)
}

$osCaption = [System.Environment]::OSVersion.VersionString
$osVersion = [System.Environment]::OSVersion.Version.ToString()
$osBuild = [System.Environment]::OSVersion.Version.Build.ToString()
$osArchitecture = $env:PROCESSOR_ARCHITECTURE
if ($operatingSystem -ne $null) {
    $osCaption = ([string]$operatingSystem.Caption).Trim()
    $osVersion = [string]$operatingSystem.Version
    $osBuild = [string]$operatingSystem.BuildNumber
    $osArchitecture = [string]$operatingSystem.OSArchitecture
}

$environment = [ordered]@{
    collected_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    computer_name = $env:COMPUTERNAME
    cpu = [ordered]@{
        names = $cpuNames
        physical_cores = $physicalCores
        logical_threads = $logicalThreads
    }
    gpu = [ordered]@{
        names = $gpuNames
    }
    npu = [ordered]@{
        names = $npuNames
        detected = [bool]($npuNames.Count -gt 0)
    }
    memory = [ordered]@{
        total_bytes = $totalMemoryBytes
        total_gib = $totalMemoryGib
    }
    operating_system = [ordered]@{
        caption = $osCaption
        version = $osVersion
        build_number = $osBuild
        architecture = $osArchitecture
    }
    power_plan = $powerPlan
}

$fullOutputPath = [System.IO.Path]::GetFullPath($OutputPath)
$parentDirectory = Split-Path -Parent $fullOutputPath
if (-not (Test-Path -LiteralPath $parentDirectory)) {
    New-Item -ItemType Directory -Path $parentDirectory -Force | Out-Null
}

$json = $environment | ConvertTo-Json -Depth 8
$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($fullOutputPath, $json, $utf8WithoutBom)
Write-Output $fullOutputPath

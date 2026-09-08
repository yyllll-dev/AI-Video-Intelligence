<#
.SYNOPSIS
Starts the VisionOracle Gradio application on Windows.

.DESCRIPTION
The script prefers .venv\Scripts\python.exe, then the active Conda/Python
environment, and finally the Windows Python launcher. It validates the runtime,
required packages, YOLO weights, and an optional local Qwen model path before
starting demo\app.py.

.EXAMPLE
.\run.ps1 -QwenModelPath "D:\AIModels\Qwen2-VL-2B-Instruct" -YoloDevice 0

.EXAMPLE
.\run.ps1 -InstallDependencies -YoloDevice cpu

.EXAMPLE
.\run.ps1 -CheckOnly
#>

[CmdletBinding()]
param(
    [string]$PythonPath = "",
    [string]$QwenModelPath = "",
    [string]$YoloDevice = "",
    [ValidateRange(0.0, 1.0)]
    [double]$YoloConfidence = 0.35,
    [string]$ServerName = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 7860,
    [switch]$InstallDependencies,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$requirementsPath = Join-Path $projectRoot "requirements.txt"
$appPath = Join-Path $projectRoot "demo\app.py"
$yoloWeightPath = Join-Path $projectRoot "models\yolo11n.pt"

function Resolve-PythonInvocation {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        $resolved = Resolve-Path -LiteralPath $RequestedPath -ErrorAction Stop
        return [PSCustomObject]@{
            Command = $resolved.Path
            PrefixArgs = @()
            Description = $resolved.Path
        }
    }

    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        return [PSCustomObject]@{
            Command = $venvPython
            PrefixArgs = @()
            Description = $venvPython
        }
    }

    if ($env:CONDA_PREFIX) {
        $condaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path -LiteralPath $condaPython -PathType Leaf) {
            return [PSCustomObject]@{
                Command = $condaPython
                PrefixArgs = @()
                Description = $condaPython
            }
        }
    }

    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    }
    if ($pythonCommand) {
        return [PSCustomObject]@{
            Command = $pythonCommand.Source
            PrefixArgs = @()
            Description = $pythonCommand.Source
        }
    }

    $pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if (-not $pyLauncher) {
        $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    }
    if ($pyLauncher) {
        return [PSCustomObject]@{
            Command = $pyLauncher.Source
            PrefixArgs = @("-3.11")
            Description = "$($pyLauncher.Source) -3.11"
        }
    }

    throw "Python was not found. Install Python 3.11, create .venv, or pass -PythonPath."
}

function Invoke-SelectedPython {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Python,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    $prefixArgs = @($Python.PrefixArgs)
    & $Python.Command @prefixArgs @Arguments
}

if (-not (Test-Path -LiteralPath $requirementsPath -PathType Leaf)) {
    throw "Requirements file not found: $requirementsPath"
}
if (-not (Test-Path -LiteralPath $appPath -PathType Leaf)) {
    throw "Application entry point not found: $appPath"
}
if (-not (Test-Path -LiteralPath $yoloWeightPath -PathType Leaf)) {
    throw "YOLO weights not found: $yoloWeightPath"
}

$python = Resolve-PythonInvocation -RequestedPath $PythonPath
$versionText = Invoke-SelectedPython -Python $python -Arguments @(
    "-c",
    "import sys; print('.'.join(map(str, sys.version_info[:3])))"
)
if ($LASTEXITCODE -ne 0) {
    throw "Python could not be started: $($python.Description)"
}

$version = [version](($versionText | Select-Object -Last 1).Trim())
if ($version.Major -ne 3 -or $version.Minor -ne 11) {
    Write-Warning "Current Python is $version. This project was verified with Python 3.11."
}

Write-Host "[VisionOracle] Project: $projectRoot"
Write-Host "[VisionOracle] Python: $($python.Description) ($version)"

if ($InstallDependencies) {
    Write-Host "[VisionOracle] Installing dependencies from requirements.txt..."
    Invoke-SelectedPython -Python $python -Arguments @(
        "-m", "pip", "install", "-r", $requirementsPath
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed. Check the PyTorch, CUDA, and network notes in README.md."
    }
}

$dependencyCheck = @'
import importlib.util

required = {
    'accelerate': 'accelerate',
    'cv2': 'opencv-python',
    'gradio': 'gradio',
    'huggingface_hub': 'huggingface-hub',
    'modelscope': 'modelscope',
    'numpy': 'numpy',
    'PIL': 'Pillow',
    'qwen_vl_utils': 'qwen-vl-utils',
    'safetensors': 'safetensors',
    'torch': 'torch',
    'torchvision': 'torchvision',
    'transformers': 'transformers',
    'ultralytics': 'ultralytics',
}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
print(', '.join(missing))
'@

$missingPackages = Invoke-SelectedPython -Python $python -Arguments @(
    "-c", $dependencyCheck
)
if ($LASTEXITCODE -ne 0) {
    throw "Python dependency validation failed."
}
$missingText = ($missingPackages -join "").Trim()
if ($missingText) {
    throw "Missing Python packages: $missingText. Run .\run.ps1 -InstallDependencies."
}

$ffmpegCheck = @'
import importlib.util
print('yes' if importlib.util.find_spec('imageio_ffmpeg') else 'no')
'@
$hasImageioFfmpeg = Invoke-SelectedPython -Python $python -Arguments @(
    "-c", $ffmpegCheck
)
if (($hasImageioFfmpeg -join "").Trim() -ne "yes" -and -not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "Neither imageio-ffmpeg nor a system FFmpeg was found. MP4 conversion and replay may fail."
}

if ($QwenModelPath) {
    $resolvedQwenPath = Resolve-Path -LiteralPath $QwenModelPath -ErrorAction Stop
    if (-not (Test-Path -LiteralPath $resolvedQwenPath.Path -PathType Container)) {
        throw "Qwen model path is not a directory: $($resolvedQwenPath.Path)"
    }
    $env:QWEN_VL_MODEL_PATH = $resolvedQwenPath.Path
} elseif ($env:QWEN_VL_MODEL_PATH) {
    if (-not (Test-Path -LiteralPath $env:QWEN_VL_MODEL_PATH -PathType Container)) {
        throw "QWEN_VL_MODEL_PATH does not point to an existing directory: $env:QWEN_VL_MODEL_PATH"
    }
}

if (-not $YoloDevice) {
    $YoloDevice = if ($env:YOLO_DEVICE) { $env:YOLO_DEVICE } else { "0" }
}

$env:YOLO_DEVICE = $YoloDevice
$env:YOLO_CONFIDENCE = $YoloConfidence.ToString(
    [System.Globalization.CultureInfo]::InvariantCulture
)
$env:GRADIO_SERVER_NAME = $ServerName
$env:GRADIO_SERVER_PORT = [string]$Port

Write-Host "[VisionOracle] YOLO device: $env:YOLO_DEVICE"
Write-Host "[VisionOracle] YOLO confidence: $env:YOLO_CONFIDENCE"
if ($env:QWEN_VL_MODEL_PATH) {
    Write-Host "[VisionOracle] Qwen model: $env:QWEN_VL_MODEL_PATH"
} else {
    Write-Host "[VisionOracle] Qwen model: ModelScope download/cache (no local path configured)"
}
Write-Host "[VisionOracle] URL: http://${ServerName}:$Port"

if ($CheckOnly) {
    Write-Host "[VisionOracle] Preflight check passed."
    exit 0
}

if (-not (Test-Path -LiteralPath (Join-Path $projectRoot "data"))) {
    New-Item -ItemType Directory -Path (Join-Path $projectRoot "data") | Out-Null
}

Push-Location $projectRoot
try {
    Invoke-SelectedPython -Python $python -Arguments @($appPath)
    exit $LASTEXITCODE
} finally {
    Pop-Location
}

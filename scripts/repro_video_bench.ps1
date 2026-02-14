<#
One-click repro for the frozen video benchmark.

Defaults:
- Uses local benchmarks clip if present, then temp cache, else downloads.
- Fetches benchmark script from GitHub main (use -UseLocalScript to override).
- Uses a local venv if present; run -Setup to create one and install deps.

Examples:
  .\repro_video_bench.ps1
  .\repro_video_bench.ps1 -Setup
  .\repro_video_bench.ps1 -Legacy
  .\repro_video_bench.ps1 -ForceDownload
  .\repro_video_bench.ps1 -UseLocalScript
#>
param(
  [switch]$Setup,
  [string]$Device = "cuda",
  [string]$Backend = "dshow",
  [int]$InputSize = 512,
  [double]$Downsample = 0.25,
  [int]$Width = 1280,
  [int]$Height = 720,
  [int]$Duration = 30,
  [double]$BlurScale = 0.5,
  [double]$BlurSigma = 8.0,
  [string]$VideoUrl = "https://www.pexels.com/download/video/6517471/",
  [string]$VideoCachePath = "$env:TEMP\\edgescope-bench-6517471.mp4",
  [switch]$ForceDownload,
  [string]$ScriptUrl = "https://raw.githubusercontent.com/mcherif/edgescope-studio/main/scripts/benchmark_video.py",
  [string]$RepoZipUrl = "https://github.com/mcherif/edgescope-studio/archive/refs/heads/main.zip",
  [string]$RepoCacheDir = "$env:TEMP\\edgescope-studio-main",
  [switch]$UseLocalScript,
  [switch]$SkipModelDownload,
  [switch]$ForceCuda,
  [switch]$Legacy
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Find-RepoRoot([string]$start) {
  $current = $start
  while ($current -and (Test-Path $current)) {
    if (Test-Path (Join-Path $current "edgescope")) { return $current }
    $parent = Split-Path -Parent $current
    if ($parent -eq $current) { break }
    $current = $parent
  }
  return $null
}

function Ensure-Repo([string]$zipUrl, [string]$cacheDir) {
  if (-not (Test-Path $cacheDir)) {
    Write-Host "Downloading repo to: $cacheDir"
    $zipPath = "$cacheDir.zip"
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath (Split-Path -Parent $cacheDir) -Force
    Remove-Item $zipPath -Force
  }
  return $cacheDir
}

function Ensure-Python {
  $py = Get-Command python -ErrorAction SilentlyContinue
  if ($py) { return }
  Write-Host "Python not found. Attempting install via winget..."
  $winget = Get-Command winget -ErrorAction SilentlyContinue
  if (-not $winget) {
    throw "Python not found and winget is unavailable. Install Python 3.10+ and retry."
  }
  winget install -e --id Python.Python.3.11
}

function Ensure-Venv([string]$root) {
  $venvDir = Join-Path $root ".venv-video"
  $venvPy = Join-Path $venvDir "Scripts/python.exe"
  if ($Setup -or -not (Test-Path $venvPy)) {
    Ensure-Python
    Write-Host "Creating venv: $venvDir"
    python -m venv $venvDir
    Write-Host "Installing deps into venv..."
    & $venvPy -m pip install --upgrade pip 2>&1 | Out-Host
    & $venvPy -m pip install onnxruntime-gpu opencv-python numpy 2>&1 | Out-Host
  }
  if (Test-Path $venvPy) { return $venvPy }
  return "python"
}

function Ensure-Model([string]$root, [string]$py) {
  $modelPath = Join-Path $root "models/rvm_mobilenetv3_fp32.onnx"
  if (Test-Path $modelPath) { return }
  if ($SkipModelDownload) {
    throw "Model missing: $modelPath (use -SkipModelDownload:$false or download models first)"
  }
  Write-Host "Downloading RVM model..."
  & $py (Join-Path $root "scripts/setup_video.py")
}

function Get-Providers([string]$py) {
  $code = "import onnxruntime as ort; print(','.join(ort.get_available_providers()))"
  return & $py -c $code
}

$cudaBase = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"
if (Test-Path $cudaBase) {
  $cudaDirs = Get-ChildItem $cudaBase -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
  if ($cudaDirs.Count -gt 0) {
    $cudaBin = Join-Path $cudaDirs[0].FullName "bin"
    if ((Test-Path $cudaBin) -and (-not $env:PATH.Contains($cudaBin))) {
      $env:PATH = "$cudaBin;$env:PATH"
    }
  }
}

function Test-DllInPath([string]$dll) {
  foreach ($p in ($env:PATH -split ';')) {
    if ($p -and (Test-Path (Join-Path $p $dll))) { return $true }
  }
  return $false
}

function Get-MissingCudaDlls {
  $required = @("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll")
  $missing = @()
  foreach ($dll in $required) {
    if (-not (Test-DllInPath $dll)) { $missing += $dll }
  }
  return $missing
}

$repoRoot = Find-RepoRoot (Split-Path -Parent $PSScriptRoot)
if (-not $repoRoot) {
  $repoRoot = Ensure-Repo $RepoZipUrl $RepoCacheDir
}
Set-Location $repoRoot

$py = Ensure-Venv $repoRoot
Write-Host "Using Python: $py"
Ensure-Model $repoRoot $py

$localBenchPath = Join-Path $repoRoot "benchmarks/6517471-hd_1920_1080_30fps.mp4"
if ((Test-Path $localBenchPath) -and (-not $ForceDownload)) {
  $videoPath = $localBenchPath
  Write-Host "Using local benchmark clip: $videoPath"
} elseif ((Test-Path $VideoCachePath) -and (-not $ForceDownload)) {
  $videoPath = $VideoCachePath
  Write-Host "Using cached benchmark clip: $videoPath"
} else {
  $videoDir = Split-Path -Parent $VideoCachePath
  if (-not (Test-Path $videoDir)) { New-Item -ItemType Directory -Path $videoDir | Out-Null }
  Write-Host "Downloading benchmark clip from: $VideoUrl"
  Invoke-WebRequest -Uri $VideoUrl -OutFile $VideoCachePath
  $videoPath = $VideoCachePath
}

$scriptDir = Join-Path $repoRoot "scripts"
if (-not (Test-Path $scriptDir)) { New-Item -ItemType Directory -Path $scriptDir | Out-Null }

$scriptPath = Join-Path $scriptDir "benchmark_video.remote.py"
if ($UseLocalScript) {
  $scriptPath = Join-Path $scriptDir "benchmark_video.py"
} else {
  Write-Host "Fetching benchmark script from: $ScriptUrl"
  Invoke-WebRequest -Uri $ScriptUrl -OutFile $scriptPath
}

$alphaPath = "optimized"
if ($Legacy) { $alphaPath = "legacy" }

$outName = "benchmarks/rvm_512_ds025_720p_blur_soft_profile_video6517471_full10s_opt.json"
if ($Legacy) { $outName = "benchmarks/rvm_512_ds025_720p_blur_soft_profile_video6517471_full10s_legacy.json" }

$deviceEffective = $Device
if ($Device -eq "cuda") {
  $providers = Get-Providers $py
  Write-Host "ONNX Runtime providers: $providers"
  $missingDlls = Get-MissingCudaDlls
  if (@($missingDlls).Count -gt 0) {
    $msg = "CUDA DLLs missing: " + ($missingDlls -join ", ")
    if ($ForceCuda) { throw $msg }
    Write-Warning "$msg. Falling back to CPU for this run."
    $deviceEffective = "cpu"
  }
  $hasCuda = $providers -match "CUDAExecutionProvider"
  if (-not $hasCuda -and $deviceEffective -eq "cuda") {
    if ($ForceCuda) {
      throw "CUDAExecutionProvider not available; install CUDA-enabled onnxruntime or use -Device cpu."
    }
    Write-Warning "CUDAExecutionProvider not available; falling back to CPU for this run."
    $deviceEffective = "cpu"
  } else {
    Write-Host "CUDAExecutionProvider available; using CUDA."
  }
}

Write-Host "Running benchmark..."
& $py $scriptPath --device $deviceEffective --backend $Backend `
  --video $videoPath --video-frame-index 0 --video-frame-count 0 `
  --input-size $InputSize --downsample $Downsample --width $Width --height $Height --duration $Duration `
  --blur --blur-scale $BlurScale --blur-sigma $BlurSigma --comp-mode soft --alpha-path $alphaPath `
  --out $outName

if ($LASTEXITCODE -ne 0) {
  throw "Benchmark failed with exit code $LASTEXITCODE"
}

Write-Host "Done. Wrote: $outName"

<#
One-click repro for the frozen video benchmark.

Defaults:
- Uses local benchmarks clip if present, then temp cache, else downloads.
- Fetches benchmark script from GitHub main (use -UseLocalScript to override).
- Uses a local venv if present; run -Setup to create one and install deps.

Examples:
  .\repro_video_bench.ps1
  .\repro_video_bench.ps1 -Setup
  .\repro_video_bench.ps1 -UseVenv
  .\repro_video_bench.ps1 -SetupNoConda
  .\repro_video_bench.ps1 -Legacy
  .\repro_video_bench.ps1 -ForceDownload
  .\repro_video_bench.ps1 -UseLocalScript
#>
param(
  [switch]$Setup,
  [switch]$UseVenv,
  [switch]$SetupNoConda,
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
  [string]$CudaPath = "",
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

function Resolve-Python([string]$root) {
  $venvDir = Join-Path $root ".venv-video"
  $venvPy = Join-Path $venvDir "Scripts/python.exe"
  if ($Setup -or $SetupNoConda -or $UseVenv) {
    Ensure-Python
    if (-not (Test-Path $venvPy)) {
      Write-Host "Creating venv: $venvDir"
      python -m venv $venvDir
    }
    if ($Setup -or $SetupNoConda) {
      Write-Host "Installing deps into venv..."
      & $venvPy -m pip install --upgrade pip 2>&1 | Out-Host
      & $venvPy -m pip install onnxruntime-gpu opencv-python numpy 2>&1 | Out-Host
      if ($SetupNoConda) {
        Write-Host "Installing CUDA runtime/cuDNN wheels (no conda)..."
        & $venvPy -m pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 2>&1 | Out-Host
      }
    }
  }
  if (($Setup -or $SetupNoConda -or $UseVenv) -and (Test-Path $venvPy)) {
    Write-Host "Using venv: $venvDir"
    return $venvPy
  }
  $py = Get-Command python -ErrorAction SilentlyContinue
  if (-not $py) {
    Ensure-Python
    $py = Get-Command python -ErrorAction SilentlyContinue
  }
  return $py.Source
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

function Use-LatestCudaBin {
  $cudaBase = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"
  if (-not (Test-Path $cudaBase)) { return }
  $cudaDirs = Get-ChildItem $cudaBase -Directory -ErrorAction SilentlyContinue
  if (-not $cudaDirs) { return }
  $versions = @()
  foreach ($d in $cudaDirs) {
    if ($d.Name -match '^v(\d+)\.(\d+)$') {
      $versions += [pscustomobject]@{
        Path = $d.FullName
        Major = [int]$Matches[1]
        Minor = [int]$Matches[2]
      }
    }
  }
  if ($versions.Count -eq 0) { return }
  $sorted = $versions | Sort-Object Major, Minor -Descending
  $selected = $null
  $selectedVer = $null
  $fallback = $null
  $fallbackVer = $null
  foreach ($v in $sorted) {
    $cudaBin = Join-Path $v.Path "bin"
    if (-not (Test-Path $cudaBin)) { continue }
    if (-not $fallback) {
      $fallback = $cudaBin
      $fallbackVer = "v$($v.Major).$($v.Minor)"
    }
    if (Test-Path (Join-Path $cudaBin "cudnn64_9.dll")) {
      $selected = $cudaBin
      $selectedVer = "v$($v.Major).$($v.Minor)"
      break
    }
  }
  if ($selected) {
    if (-not $env:PATH.Contains($selected)) {
      $env:PATH = "$selected;$env:PATH"
    }
    Write-Host "Using CUDA install: $selectedVer ($selected)"
  } elseif ($fallback) {
    if (-not $env:PATH.Contains($fallback)) {
      $env:PATH = "$fallback;$env:PATH"
    }
    Write-Warning "No CUDA bin with cudnn64_9.dll found; using $fallbackVer ($fallback)"
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

Use-LatestCudaBin

$condaBin = $null
if ($env:CONDA_PREFIX) {
  $condaBin = Join-Path $env:CONDA_PREFIX "Library\\bin"
}
if ($condaBin -and (Test-Path $condaBin) -and (-not $env:PATH.Contains($condaBin))) {
  $env:PATH = "$condaBin;$env:PATH"
}
if ($CudaPath) {
  $paths = $CudaPath -split ';'
  foreach ($p in $paths) {
    $trim = $p.Trim()
    if ($trim -eq "") { continue }
    if (-not (Test-Path $trim)) {
      Write-Warning "CudaPath not found: $trim"
      continue
    }
    if (-not $env:PATH.Contains($trim)) {
      $env:PATH = "$trim;$env:PATH"
    }
  }
}

$py = Resolve-Python $repoRoot
Write-Host "Using Python: $py"

if ($SetupNoConda) {
  $sitePackages = & $py -c "import site; print(site.getsitepackages()[0])"
  $cudnnBin = Join-Path $sitePackages "nvidia\\cudnn\\bin"
  $cublasBin = Join-Path $sitePackages "nvidia\\cublas\\bin"
  $cudaRtBin = Join-Path $sitePackages "nvidia\\cuda_runtime\\bin"
  foreach ($p in @($cudnnBin, $cublasBin, $cudaRtBin)) {
    if (Test-Path $p -and (-not $env:PATH.Contains($p))) {
      $env:PATH = "$p;$env:PATH"
    }
  }
}
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
    if (-not $env:CONDA_PREFIX -and -not $CudaPath) {
      $msg += ". If you have a CUDA conda env, run this script from that env, or pass -CudaPath `<path-to-cuda-bin>` (cuDNN bin)."
    }
    Write-Warning "$msg. Falling back to CPU for this run."
    $deviceEffective = "cpu"
  }
  if ($deviceEffective -eq "cuda") {
    $hasCuda = $providers -match "CUDAExecutionProvider"
    if (-not $hasCuda) {
      if ($ForceCuda) {
        throw "CUDAExecutionProvider not available; install CUDA-enabled onnxruntime or use -Device cpu."
      }
      Write-Warning "CUDAExecutionProvider not available; falling back to CPU for this run."
      $deviceEffective = "cpu"
    } else {
      Write-Host "CUDAExecutionProvider available; using CUDA."
    }
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

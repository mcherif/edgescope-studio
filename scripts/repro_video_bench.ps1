<#
One-click repro for the frozen video benchmark.

Defaults:
- Uses local benchmarks clip if present, then temp cache, else downloads.
- Fetches benchmark script from GitHub main (use -UseLocalScript to override).
- Uses a local venv if present; run -Setup to create one, install deps, and CUDA runtime wheels.

Examples:
  .\repro_video_bench.ps1
  .\repro_video_bench.ps1 -Setup
  .\repro_video_bench.ps1 -UseVenv
  .\repro_video_bench.ps1 -Legacy
  .\repro_video_bench.ps1 -ForceDownload
  .\repro_video_bench.ps1 -UseLocalScript
#>
param(
  [switch]$Setup,
  [switch]$UseVenv,
  [bool]$CleanPath = $true,
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
$global:VenvDllDirs = $null

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
  $resp = Read-Host "Install Python 3.11 via winget? (y/N)"
  if ($resp -notin @("y","Y")) { throw "Aborted Python install." }
  winget install -e --id Python.Python.3.11
}

function Resolve-Python([string]$root) {
  $venvDir = Join-Path $root ".venv-video"
  $venvPy = Join-Path $venvDir "Scripts/python.exe"
  if ($Setup -or $UseVenv) {
    Ensure-Python
    if (-not (Test-Path $venvPy)) {
      Write-Host "Creating venv: $venvDir"
      $resp = Read-Host "Create venv at $venvDir? (y/N)"
      if ($resp -notin @("y","Y")) { throw "Aborted venv creation." }
      python -m venv $venvDir
    }
    if ($Setup) {
      Write-Host "Installing deps into venv..."
      $resp = Read-Host "Install Python deps into venv? (y/N)"
      if ($resp -notin @("y","Y")) { throw "Aborted dependency install." }
      & $venvPy -m pip install --upgrade pip 2>&1 | Out-Host
      & $venvPy -m pip install onnxruntime-gpu opencv-python numpy 2>&1 | Out-Host
      Write-Host "Installing CUDA runtime/cuDNN wheels (no conda)..."
      $resp = Read-Host "Install CUDA runtime/cuDNN wheels? (y/N)"
      if ($resp -notin @("y","Y")) { throw "Aborted CUDA wheel install." }
      $wheelDirs = @()
      $localWheelDir = Get-Location
      if ($localWheelDir) { $wheelDirs += $localWheelDir.Path }
      $pipCacheDir = (& $venvPy -m pip cache dir) 2>$null
      if ($pipCacheDir -and (Test-Path $pipCacheDir)) { $wheelDirs += $pipCacheDir.Trim() }
      $wheelNames = @(
        "nvidia_cuda_runtime_cu12",
        "nvidia_cublas_cu12",
        "nvidia_cufft_cu12",
        "nvidia_cuda_nvrtc_cu12",
        "nvidia_cudnn_cu12"
      )
      $foundWheelDirs = @()
      foreach ($d in ($wheelDirs | Select-Object -Unique)) {
        $hits = @()
        foreach ($n in $wheelNames) {
          $pattern = "$n-*.whl"
          $hits += Get-ChildItem -Path $d -Recurse -Filter $pattern -ErrorAction SilentlyContinue
        }
        if ($hits.Count -gt 0) { $foundWheelDirs += $d }
      }
      if ($foundWheelDirs.Count -gt 0) {
        $dirsArg = $foundWheelDirs | Select-Object -Unique
        Write-Host "Found CUDA wheels locally/cache: $($dirsArg -join ', ')"
        & $venvPy -m pip install --no-index --find-links $dirsArg `
          nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cudnn-cu12 2>&1 | Out-Host
      } else {
        Write-Host "No CUDA wheels found locally/cache; downloading from PyPI..."
        & $venvPy -m pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cudnn-cu12 2>&1 | Out-Host
      }
    }
  }
  if (($Setup -or $UseVenv) -and (Test-Path $venvPy)) {
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

function Get-VenvContext([string]$py) {
  $scriptsDir = Split-Path -Parent $py
  $venvRoot = Split-Path -Parent $scriptsDir
  $sitePackages = (& $py -c "import site; print(site.getsitepackages()[0])") 2>$null
  $nvidiaBins = @(
    (Join-Path $sitePackages "nvidia\\cuda_runtime\\bin"),
    (Join-Path $sitePackages "nvidia\\cublas\\bin"),
    (Join-Path $sitePackages "nvidia\\cufft\\bin"),
    (Join-Path $sitePackages "nvidia\\cuda_nvrtc\\bin"),
    (Join-Path $sitePackages "nvidia\\cudnn\\bin")
  ) | Where-Object { $_ -and (Test-Path $_) }
  return [pscustomobject]@{
    ScriptsDir = $scriptsDir
    VenvRoot = $venvRoot
    SitePackages = $sitePackages
    NvidiaBins = $nvidiaBins
  }
}

function Set-CleanPathForVenv([string]$py) {
  $ctx = Get-VenvContext $py
  $prepend = @($ctx.ScriptsDir) + $ctx.NvidiaBins
  $current = $env:PATH -split ';' | Where-Object { $_ -and $_.Trim() -ne "" }
  $filtered = @()
  foreach ($p in $current) {
    $lp = $p.ToLowerInvariant()
    if ($env:CONDA_PREFIX -and $p.StartsWith($env:CONDA_PREFIX, [System.StringComparison]::OrdinalIgnoreCase)) {
      continue
    }
    if ($lp.Contains("\\miniconda") -or $lp.Contains("\\anaconda") -or $lp.Contains("\\conda")) {
      continue
    }
    $filtered += $p
  }
  $merged = @()
  foreach ($p in ($prepend + $filtered)) {
    if (-not $p) { continue }
    if (-not ($merged -contains $p)) { $merged += $p }
  }
  $env:PATH = ($merged -join ';')
  Write-Host "Clean PATH mode enabled: conda paths removed, venv paths prepended."
  Write-Host ("Prepended: " + (($prepend | Where-Object { $_ }) -join ", "))
}

function Log-RuntimeProbe([string]$py, [string]$modelPath) {
  $probeCode = @'
import ctypes
import json
import os
from pathlib import Path
import onnxruntime as ort

def in_path(name):
    for p in os.environ.get("PATH", "").split(";"):
        if not p:
            continue
        c = Path(p) / name
        if c.exists():
            return str(c)
    return None

def loaded_module(name):
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    k32.GetModuleHandleW.restype = ctypes.c_void_p
    h = k32.GetModuleHandleW(name)
    if not h:
        return None
    buf = ctypes.create_unicode_buffer(4096)
    k32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
    k32.GetModuleFileNameW.restype = ctypes.c_uint
    n = k32.GetModuleFileNameW(h, buf, len(buf))
    return buf.value if n else None

model = os.environ.get("REPRO_MODEL_PATH")
probe = {
    "sys_executable": os.path.abspath(os.sys.executable),
    "ort_version": ort.__version__,
    "providers_available": ort.get_available_providers(),
    "session_providers": None,
    "session_error": None,
    "dlls": {
        "onnxruntime_providers_cuda.dll": {
            "resolved_path": str((Path(ort.__file__).resolve().parent / "capi" / "onnxruntime_providers_cuda.dll"))
        },
        "cudnn64_9.dll": {"resolved_path": in_path("cudnn64_9.dll"), "loaded_module_path": loaded_module("cudnn64_9.dll")},
        "cublas64_12.dll": {"resolved_path": in_path("cublas64_12.dll"), "loaded_module_path": loaded_module("cublas64_12.dll")},
        "cudart64_12.dll": {"resolved_path": in_path("cudart64_12.dll"), "loaded_module_path": loaded_module("cudart64_12.dll")},
    },
}
if model and Path(model).exists():
    try:
        sess = ort.InferenceSession(model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        probe["session_providers"] = sess.get_providers()
    except Exception as e:
        probe["session_error"] = str(e)
print("Runtime probe:")
print(json.dumps(probe, indent=2))
'@
  $env:REPRO_MODEL_PATH = $modelPath
  $probeCode | & $py -
}

function Ensure-VenvDllHook([string]$py) {
  if (-not ($py -like "*\.venv-video*")) { return }
  if (-not $global:VenvDllDirs -or $global:VenvDllDirs.Count -eq 0) { return }
  $scriptsDir = Split-Path -Parent $py
  $venvRoot = Split-Path -Parent $scriptsDir
  $dllHackDir = Join-Path $venvRoot ".dll-hacks"
  if (-not (Test-Path $dllHackDir)) { New-Item -ItemType Directory -Path $dllHackDir | Out-Null }
  $dllList = ($global:VenvDllDirs -join ";")
  $siteCustomize = @"
import os
dirs = r"$dllList".split(";")
for d in dirs:
    if d:
        try:
            os.add_dll_directory(d)
        except Exception:
            pass
"@
  Set-Content -Path (Join-Path $dllHackDir "sitecustomize.py") -Value $siteCustomize -Encoding ASCII
  if ($env:PYTHONPATH) {
    $parts = $env:PYTHONPATH -split ';' | Where-Object { $_ -and $_.Trim() -ne "" }
    if (-not ($parts -contains $dllHackDir)) {
      $env:PYTHONPATH = "$dllHackDir;$env:PYTHONPATH"
    }
  } else {
    $env:PYTHONPATH = $dllHackDir
  }
  Write-Host "Injected sitecustomize to add DLL dirs for CUDA."
}

function Add-VenvCudaBins([string]$py) {
  $dirs = (& $py -c "import site,os; sp=site.getsitepackages()[0]; dirs=[os.path.join(sp,'nvidia',n,'bin') for n in ('cudnn','cublas','cufft','cuda_runtime','cuda_nvrtc')]; print(';'.join([d for d in dirs if os.path.isdir(d)]))") 2>$null
  $added = @()
  $dirList = @()
  if ($dirs) { $dirList = $dirs -split ';' }
  if ($dirList.Count -gt 0) {
    Write-Host ("Venv NVIDIA bin dirs: " + ($dirList -join ", "))
  } else {
    Write-Host "Venv NVIDIA bin dirs: (none found)"
  }
  if ($dirs) {
    foreach ($d in ($dirs -split ';')) {
      if ($d -and (-not $env:PATH.Contains($d))) {
        $env:PATH = "$d;$env:PATH"
        $added += $d
      }
    }
  }
  $dllDirs = (& $py -c "import site,glob,os; sp=site.getsitepackages()[0]; req=['cudnn64_9.dll','cublasLt64_12.dll','cublas64_12.dll','cudart64_12.dll','cufft64_11.dll','nvrtc64_120_0.dll']; dirs=set(); [dirs.add(os.path.dirname(ms[0])) for r in req for ms in [glob.glob(os.path.join(sp,'**',r), recursive=True)] if ms]; print(';'.join(dirs))") 2>$null
  $dllDirList = @()
  if ($dllDirs) { $dllDirList = $dllDirs -split ';' }
  if ($dllDirList.Count -gt 0) {
    Write-Host ("Venv CUDA DLL dirs: " + ($dllDirList -join ", "))
  } else {
    Write-Host "Venv CUDA DLL dirs: (none found)"
  }
  if ($dllDirs) {
    foreach ($d in ($dllDirs -split ';')) {
      if ($d -and (-not $env:PATH.Contains($d))) {
        $env:PATH = "$d;$env:PATH"
        $added += $d
      }
    }
  }
  if ($added.Count -gt 0) {
    $uniqueAdded = $added | Select-Object -Unique
    Write-Host ("Added CUDA DLL dirs from venv: " + ($uniqueAdded -join ", "))
  }
  if ($dllDirList.Count -gt 0) {
    $global:VenvDllDirs = $dllDirList | Select-Object -Unique
  }
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
  $required = @("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cufft64_11.dll", "nvrtc64_120_0.dll", "cudnn64_9.dll")
  $missing = @()
  foreach ($dll in $required) {
    if (-not (Test-DllInPath $dll)) { $missing += $dll }
  }
  return $missing
}

function Debug-CudaDlls {
  $required = @("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cufft64_11.dll", "nvrtc64_120_0.dll", "cudnn64_9.dll")
  foreach ($dll in $required) {
    if (Test-DllInPath $dll) {
      Write-Host "Found in PATH: $dll"
    } else {
      Write-Host "Missing in PATH: $dll"
    }
  }
}

$repoRoot = Find-RepoRoot (Split-Path -Parent $PSScriptRoot)
if (-not $repoRoot) {
  $resp = Read-Host "Download repo to ${RepoCacheDir}? (y/N)"
  if ($resp -notin @("y","Y")) { throw "Aborted repo download." }
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

if ($py -and ($py -like "*\.venv-video*")) {
  if ($CleanPath) {
    Set-CleanPathForVenv $py
  }
  Add-VenvCudaBins $py
  Ensure-VenvDllHook $py
}
Debug-CudaDlls
Ensure-Model $repoRoot $py
$modelPath = Join-Path $repoRoot "models/rvm_mobilenetv3_fp32.onnx"
Log-RuntimeProbe $py $modelPath

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
  $resp = Read-Host "Download benchmark clip? (y/N)"
  if ($resp -notin @("y","Y")) { throw "Aborted video download." }
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
  $resp = Read-Host "Download benchmark script? (y/N)"
  if ($resp -notin @("y","Y")) { throw "Aborted script download." }
  Invoke-WebRequest -Uri $ScriptUrl -OutFile $scriptPath
}

$alphaPath = "optimized"
if ($Legacy) { $alphaPath = "legacy" }

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outName = "benchmarks/rvm_512_ds025_720p_blur_soft_profile_video6517471_full10s_opt_ref_${stamp}.json"
if ($Legacy) { $outName = "benchmarks/rvm_512_ds025_720p_blur_soft_profile_video6517471_full10s_legacy_ref_${stamp}.json" }

$deviceEffective = $Device
if ($Device -eq "cuda") {
  $providers = Get-Providers $py
  Write-Host "ONNX Runtime providers: $providers"
  if (($providers -notmatch "CUDAExecutionProvider") -and (-not ($py -like "*\\.venv-video*"))) {
    $venvDir = Join-Path $repoRoot ".venv-video"
    $venvPy = Join-Path $venvDir "Scripts/python.exe"
    if (Test-Path $venvPy) {
      $resp = Read-Host "CUDA not available in current Python. Use venv at ${venvDir}? (y/N)"
      if ($resp -in @("y","Y")) {
        $py = $venvPy
        Write-Host "Switched to venv Python: $py"
        if ($CleanPath) {
          Set-CleanPathForVenv $py
        }
        Add-VenvCudaBins $py
        Ensure-VenvDllHook $py
        Debug-CudaDlls
        Log-RuntimeProbe $py $modelPath
        $providers = Get-Providers $py
        Write-Host "ONNX Runtime providers (venv): $providers"
      }
    }
  }
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

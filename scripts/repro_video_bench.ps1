param(
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
  [switch]$UseCachedVideo,
  [string]$ScriptUrl = "https://raw.githubusercontent.com/mcherif/edgescope-studio/main/scripts/benchmark_video.py",
  [string]$RepoZipUrl = "https://github.com/mcherif/edgescope-studio/archive/refs/heads/main.zip",
  [string]$RepoCacheDir = "$env:TEMP\\edgescope-studio-main",
  [switch]$UseLocalScript,
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

$repoRoot = Find-RepoRoot (Split-Path -Parent $PSScriptRoot)
if (-not $repoRoot) {
  $repoRoot = Ensure-Repo $RepoZipUrl $RepoCacheDir
}
Set-Location $repoRoot

if ($UseCachedVideo -and (Test-Path $VideoCachePath)) {
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

Write-Host "Running benchmark..."
python $scriptPath --device $Device --backend $Backend `
  --video $videoPath --video-frame-index 0 --video-frame-count 0 `
  --input-size $InputSize --downsample $Downsample --width $Width --height $Height --duration $Duration `
  --blur --blur-scale $BlurScale --blur-sigma $BlurSigma --comp-mode soft --alpha-path $alphaPath `
  --out $outName

if ($LASTEXITCODE -ne 0) {
  throw "Benchmark failed with exit code $LASTEXITCODE"
}

Write-Host "Done. Wrote: $outName"

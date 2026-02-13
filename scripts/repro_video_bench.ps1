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
  [string]$ScriptUrl = "https://raw.githubusercontent.com/mcherif/edgescope-studio/main/scripts/benchmark_video.py",
  [switch]$UseLocalScript,
  [switch]$Legacy
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "Streaming benchmark clip from: $VideoUrl"

$scriptPath = Join-Path $PSScriptRoot "benchmark_video.remote.py"
if ($UseLocalScript) {
  $scriptPath = Join-Path $PSScriptRoot "benchmark_video.py"
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
  --video $VideoUrl --video-frame-index 0 --video-frame-count 0 `
  --input-size $InputSize --downsample $Downsample --width $Width --height $Height --duration $Duration `
  --blur --blur-scale $BlurScale --blur-sigma $BlurSigma --comp-mode soft --alpha-path $alphaPath `
  --out $outName

Write-Host "Done. Wrote: $outName"

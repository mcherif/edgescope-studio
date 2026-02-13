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
  [switch]$Legacy,
  [switch]$ForceDownload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$videoUrl = "https://www.pexels.com/download/video/6517471/"
$videoPath = "benchmarks/6517471-hd_1920_1080_30fps.mp4"
$shaPath = "benchmarks/6517471-hd_1920_1080_30fps.sha256"

function Get-ExpectedSha256([string]$path) {
  if (-not (Test-Path $path)) { return $null }
  $line = (Get-Content -Encoding utf8 $path | Select-Object -First 1).Trim()
  if ($line -eq "") { return $null }
  return ($line -split "\s+")[0]
}

function Get-ActualSha256([string]$path) {
  if (-not (Test-Path $path)) { return $null }
  return (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower()
}

if ($ForceDownload -or -not (Test-Path $videoPath)) {
  Write-Host "Downloading benchmark clip..."
  python scripts/fetch_benchmark_video.py --url $videoUrl --out $videoPath --sha-out $shaPath --force
} else {
  $expected = Get-ExpectedSha256 $shaPath
  if ($expected) {
    $actual = Get-ActualSha256 $videoPath
    if ($actual -and ($actual -ne $expected.ToLower())) {
      Write-Warning "SHA256 mismatch; re-downloading."
      python scripts/fetch_benchmark_video.py --url $videoUrl --out $videoPath --sha-out $shaPath --force
    } else {
      Write-Host "Using existing clip (SHA256 OK)."
    }
  } else {
    Write-Host "Using existing clip (no SHA256 to verify)."
  }
}

$alphaPath = "optimized"
if ($Legacy) { $alphaPath = "legacy" }

$outName = "benchmarks/rvm_512_ds025_720p_blur_soft_profile_video6517471_full10s_opt.json"
if ($Legacy) { $outName = "benchmarks/rvm_512_ds025_720p_blur_soft_profile_video6517471_full10s_legacy.json" }

Write-Host "Running benchmark..."
python scripts/benchmark_video.py --device $Device --backend $Backend `
  --video $videoPath --video-frame-index 0 --video-frame-count 0 `
  --input-size $InputSize --downsample $Downsample --width $Width --height $Height --duration $Duration `
  --blur --blur-scale $BlurScale --blur-sigma $BlurSigma --comp-mode soft --alpha-path $alphaPath `
  --out $outName

Write-Host "Done. Wrote: $outName"

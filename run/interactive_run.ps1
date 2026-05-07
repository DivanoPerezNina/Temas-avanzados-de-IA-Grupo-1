Param(
    [string]$Dataset = "AML-Small-HI",
    [string]$CfgSuffix = "GINE",
    [string]$CfgDir = "configs",
    [string]$DataDir = "./data",
    [string]$OutDirBase = "./results",
    [int]$Repeat = 1,
    [int]$Gpu = -1,
    [switch]$DisableWandb
)

# Run this script from the project root dir.

$cfgFile = Join-Path $CfgDir "$Dataset/$Dataset-$CfgSuffix.yaml"
if (-not (Test-Path $cfgFile)) {
    Write-Host "WARNING: Config does not exist: $cfgFile"
    exit 1
}

$outDir = Join-Path $OutDirBase $Dataset
$cmd = @(
    "-m", "fraudGT.main",
    "--cfg", $cfgFile,
    "--repeat", $Repeat,
    "--gpu", $Gpu,
    "out_dir", $outDir,
    "dataset.dir", $DataDir
)
if ($DisableWandb) {
    $cmd += @("wandb.use", "False")
}

Write-Host "Run program: python $($cmd -join ' ')"
Write-Host "  output dir: $outDir"

python @cmd

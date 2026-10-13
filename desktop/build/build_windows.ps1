# Build the Windows desktop app (run by build.sh from WSL, or directly on Windows
# after staging the sources in $Work\src).
#   PyInstaller -> self-test of the bundle -> zip -> size report
param(
  [string]$Work = "$env:LOCALAPPDATA\latticegen-build",
  [string]$Python = "$env:USERPROFILE\miniconda3\envs\latticegen-build\python.exe",
  [string]$Version = "dev"
)
$ErrorActionPreference = "Stop"
Set-Location $Work
# what `conda activate` would add: lets PyInstaller find conda's DLLs (e.g. libmpdec for _decimal)
$envDir = Split-Path $Python
$env:PATH = "$envDir;$envDir\Library\bin;$envDir\DLLs;$env:PATH"

& $Python -m PyInstaller --noconfirm --clean --log-level WARN `
  --distpath "$Work\dist" --workpath "$Work\work" "$Work\src\desktop\build\latticegen.spec"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$app = "$Work\dist\LatticeGenerator"
Write-Host "`n== self-test of the bundle"
& "$app\latticegen.exe" selftest "$Work\selftest.txt"
$selftest = $LASTEXITCODE

$zip = "$Work\LatticeGenerator-win64-$Version.zip"
Remove-Item $zip -ErrorAction SilentlyContinue
Push-Location "$Work\dist"
& $Python -c "import shutil; shutil.make_archive(r'$($zip -replace '\.zip$','')', 'zip', '.', 'LatticeGenerator')"
Pop-Location

$unzipped = (Get-ChildItem $app -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
$zipped = (Get-Item $zip).Length / 1MB
Write-Host ("`n== size: {0:N0} MB unzipped, {1:N0} MB zip  ({2})" -f $unzipped, $zipped, $zip)
Write-Host "largest parts:"
Get-ChildItem "$app\_internal" | ForEach-Object {
  $s = if ($_.PSIsContainer) { (Get-ChildItem $_.FullName -Recurse -File | Measure-Object Length -Sum).Sum } else { $_.Length }
  [pscustomobject]@{ MB = [math]::Round($s / 1MB, 1); Name = $_.Name }
} | Sort-Object MB -Descending | Select-Object -First 12 | Format-Table -AutoSize | Out-String | Write-Host
exit $selftest

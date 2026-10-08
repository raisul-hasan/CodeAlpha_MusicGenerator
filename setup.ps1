param(
    [switch]$ReuseInstalledPackages,
    [ValidateSet('cpu', 'cu126')][string]$TorchBuild = 'cpu'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if ($ReuseInstalledPackages) {
        python -m venv --system-site-packages .venv
    } else {
        python -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $projectPython -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('torch') else 1)"
if ($LASTEXITCODE -ne 0) {
    & $projectPython -m pip install 'torch>=2.9,<3' --index-url "https://download.pytorch.org/whl/$TorchBuild"
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
}
& $projectPython -m pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Output 'Setup complete. Run .\start.ps1 to open the local music studio.'

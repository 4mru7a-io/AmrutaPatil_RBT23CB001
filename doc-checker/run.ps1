# Launch the Streamlit app with a Python environment that has the dependencies.
#
# Uses, in order:
#   1. $env:GENAI_PYTHON, if set
#   2. .venv\Scripts\python.exe next to this script
#   3. C:\genai-mse\.venv\Scripts\python.exe (the prepared environment on this machine)
#
# Usage: .\run.ps1

$ErrorActionPreference = "Stop"

$app = Join-Path $PSScriptRoot "app.py"
$candidates = @()
if ($env:GENAI_PYTHON) { $candidates += $env:GENAI_PYTHON }
$candidates += (Join-Path $PSScriptRoot ".venv\Scripts\python.exe")
$candidates += "C:\genai-mse\.venv\Scripts\python.exe"

$python = $null
$probe = "import importlib.util as u, sys; sys.exit(0 if all(u.find_spec(m) for m in ['streamlit', 'sentence_transformers', 'pypdf', 'docx']) else 1)"
foreach ($candidate in $candidates) {
    if (-not (Test-Path $candidate)) { continue }
    & $candidate -c $probe 2>$null
    if ($LASTEXITCODE -eq 0) {
        $python = $candidate
        break
    }
}

if (-not $python) {
    Write-Error @"
No Python environment with the required packages was found.
Tried:
$($candidates -join "`n")

Create one with:
  uv venv --python 3.12 .venv
  uv pip install --python .venv\Scripts\python.exe -r requirements.txt

Or point GENAI_PYTHON at an existing python.exe.
"@
    exit 1
}

Write-Host "Using $python"
& $python -m streamlit run $app @args

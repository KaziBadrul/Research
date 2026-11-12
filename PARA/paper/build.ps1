# Build paper/build/main.pdf: pdflatex, bibtex, pdflatex x2. Run from anywhere: powershell -File paper/build.ps1
$ErrorActionPreference = "Stop"
$bin = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64"
if (-not (Test-Path "$bin\pdflatex.exe")) { $bin = Split-Path (Get-Command pdflatex).Source }
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force build | Out-Null
function Run-Latex { & "$bin\pdflatex.exe" -interaction=nonstopmode -halt-on-error -output-directory=build main.tex | Out-Null
                     if ($LASTEXITCODE -ne 0) { Get-Content build\main.log -Tail 40; throw "pdflatex failed" } }
Run-Latex
& "$bin\bibtex.exe" build\main | Out-Host
if ($LASTEXITCODE -gt 1) { throw "bibtex failed" }
Run-Latex
Run-Latex
$log = Get-Content build\main.log
"--- warnings ---"
$log | Select-String -Pattern "undefined|Overfull|Underfull|Citation|multiply defined|Rerun" | Select-Object -ExpandProperty Line
"--- built: $(Resolve-Path build\main.pdf) ---"

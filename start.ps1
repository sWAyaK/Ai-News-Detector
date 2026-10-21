$python = Join-Path $PSScriptRoot "..\Scripts\python.exe"
Set-Location $PSScriptRoot
Write-Host "`n  NewsDetector Web App -> http://localhost:5000`n" -ForegroundColor Cyan
& $python web_app.py
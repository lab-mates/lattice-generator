@echo off
rem Starts the Lattice Generator in your browser (no installation needed).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\serve.ps1"
if errorlevel 1 pause

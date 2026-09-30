@echo off
mkdir C:\Qw3Computer 2>nul
copy /Y C:\OEM\bootstrap.ps1 C:\Qw3Computer\bootstrap.ps1
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v Qw3ComputerTest /t REG_SZ /d "powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -WindowStyle Hidden -File C:\Qw3Computer\bootstrap.ps1" /f
start "QW3 Computer Test" powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -WindowStyle Hidden -File C:\Qw3Computer\bootstrap.ps1

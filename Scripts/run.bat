@echo off
echo ===== PCB检测项目运行脚本 =====

echo 检查.NET 8.0运行时...
dotnet --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 未找到.NET运行时，请先安装.NET 8.0
    pause
    exit /b 1
)

echo 运行PCB检测程序...
dotnet run

echo ===== 程序结束 =====
pause 
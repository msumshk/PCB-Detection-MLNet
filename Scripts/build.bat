@echo off
echo ===== PCB检测项目构建脚本 =====

echo 1. 清理项目...
dotnet clean

echo 2. 恢复NuGet包...
dotnet restore

echo 3. 构建项目...
dotnet build --configuration Release

if %ERRORLEVEL% EQU 0 (
    echo 构建成功！
    echo 可执行文件位置: bin\Release\net8.0\PCBDetection.exe
) else (
    echo 构建失败！
    exit /b 1
)

echo ===== 构建完成 =====
pause 
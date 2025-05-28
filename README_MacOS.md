# PCB 缺陷检测 - MacOS 版本

这是 PCB 缺陷检测项目的 MacOS 优化版本，专门针对 macOS 系统（特别是 Apple Silicon Mac）进行了优化。

## 系统要求

- macOS 10.15 或更高版本
- .NET 8.0 SDK
- 至少 8GB RAM（推荐 16GB）
- 对于 Apple Silicon Mac：支持原生 arm64 架构

## 安装指南

### 1. 安装.NET 8.0 SDK

```bash
# 下载并安装.NET 8.0 SDK for macOS
# 访问: https://dotnet.microsoft.com/download/dotnet/8.0
# 或使用Homebrew:
brew install --cask dotnet
```

### 2. 克隆项目并切换到 MacOS 分支

```bash
git clone <repository-url>
cd PCB-Detection-MLNet
git checkout macos
```

### 3. 恢复依赖包

```bash
# 强制恢复所有包
dotnet restore --force

# 如果遇到TensorFlow相关问题，可以尝试：
dotnet clean
dotnet restore --no-cache
```

## MacOS 特定优化

### Apple Silicon 支持

本版本针对 Apple Silicon (M1/M2/M3) Mac 进行了特别优化：

1. **TensorFlow 版本升级**：使用支持 Apple Silicon 的 TensorFlow 2.16.0
2. **架构选择**：自动检测系统架构并选择合适的深度学习模型架构
3. **强制深度学习**：仅支持TensorFlow深度学习，不提供传统机器学习备用方案
4. **MacOS优化**：针对Apple Silicon和Intel Mac的特定优化设置

### 项目文件更改

- 升级`SciSharp.TensorFlow.Redist`到 2.16.0 版本
- 添加 Apple Silicon 特定的 TensorFlow 运行时包
- 增强错误处理和系统兼容性检测

## 运行项目

### 基本运行

```bash
dotnet run
```

### 指定运行时标识符（可选）

```bash
# 对于Apple Silicon Mac
dotnet run --runtime osx-arm64

# 对于Intel Mac
dotnet run --runtime osx-x64
```

## 故障排除

### TensorFlow 初始化失败

如果遇到 TensorFlow 绑定错误：

1. **检查系统架构**：

   ```bash
   uname -m  # 应该显示 arm64 (Apple Silicon) 或 x86_64 (Intel)
   ```

2. **清理并重新构建**：

   ```bash
   dotnet clean
   dotnet restore --force
   dotnet build
   ```

3. **使用 Rosetta 2（仅限 Apple Silicon）**：

   ```bash
   arch -x86_64 dotnet run
   ```

4. **强制要求**：项目仅支持TensorFlow深度学习，如果TensorFlow不可用将终止运行

### 内存不足

如果训练过程中遇到内存不足：

1. 减少批次大小
2. 使用更轻量的模型架构（ResnetV250 而不是 ResnetV2101）
3. 关闭其他应用程序释放内存

### 权限问题

确保有足够的权限访问数据文件：

```bash
chmod -R 755 PCB_detect_6_700_yolo/
```

## 性能优化建议

### Apple Silicon Mac

1. **使用原生架构**：确保运行在 arm64 模式下以获得最佳性能
2. **内存管理**：Apple Silicon 的统一内存架构可以更好地处理大型数据集
3. **温度控制**：长时间训练时注意散热

### Intel Mac

1. **使用 x64 架构**：确保使用正确的运行时标识符
2. **CPU 优化**：可能需要更多时间进行训练

## 数据集要求

确保数据集结构正确：

```
PCB_detect_6_700_yolo/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## 输出文件

训练完成后，模型文件将保存在：

- `Output/pcb_detection_model.zip` - 训练好的模型文件

## 技术特性

### 自动系统检测

- 自动检测 Apple Silicon vs Intel 架构
- 根据系统能力选择合适的训练方法
- 智能错误处理和恢复

### 纯TensorFlow深度学习

- **仅支持深度学习**：专门使用 TensorFlow ImageClassification 训练器
- **无传统机器学习备用方案**：确保使用最先进的深度学习技术
- **强制TensorFlow依赖**：必须正确配置TensorFlow环境才能运行

### 中文本地化

- 所有输出信息都有中文显示
- 错误信息和建议都使用中文
- 支持中英文缺陷类型映射

## 支持的缺陷类型

- Missing_hole - 缺孔
- Mouse_bite - 鼠咬
- Open_circuit - 开路
- Short - 短路
- Spur - 毛刺
- Spurious_copper - 多余铜箔

## 联系支持

如果遇到 MacOS 特定的问题，请提供以下信息：

1. macOS 版本：`sw_vers`
2. 系统架构：`uname -m`
3. .NET 版本：`dotnet --version`
4. 错误日志和堆栈跟踪

## 更新日志

### v1.0-macos

- 初始 MacOS 优化版本
- 支持 Apple Silicon
- 添加备用训练方案
- 中文本地化
- 增强错误处理

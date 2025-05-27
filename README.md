# PCB缺陷检测 - ML.NET深度学习训练项目

> **作者**: MsuMshk (heaize404@163.com)  
> **最后更新**: 2024年12月  
> **项目状态**: 已清理和优化，专注于PCB检测核心功能

这是一个使用ML.NET 3.0进行PCB（印刷电路板）缺陷检测的深度学习项目。项目采用ImageClassification训练器和ResNet架构，能够自动识别PCB上的6种常见缺陷类型。

## 项目特点

- ✅ **深度学习**: 使用ML.NET 3.0的ImageClassification训练器
- ✅ **ResNet架构**: 采用ResNet-V2-101预训练模型
- ✅ **YOLO数据格式**: 支持标准YOLO格式的数据集
- ✅ **完整注释**: 所有代码都有详细的中文注释和文档说明
- ✅ **模块化设计**: 清晰的项目结构，易于维护和扩展
- ✅ **多功能界面**: 支持训练、测试、单图预测等功能

## 项目结构

```
PCBDetection/
├── PCB_detect_6_700_yolo/          # 数据集目录（YOLO格式）
│   ├── train/                      # 训练数据（1660张图片）
│   │   ├── images/                 # 训练图片
│   │   └── labels/                 # 训练标签（YOLO格式）
│   ├── valid/                      # 验证数据（397张图片）
│   │   ├── images/                 # 验证图片
│   │   └── labels/                 # 验证标签
│   ├── test/                       # 测试数据（238张图片）
│   │   ├── images/                 # 测试图片
│   │   └── labels/                 # 测试标签
│   └── data.yaml                   # 数据集配置文件
├── Models/                         # 数据模型类
│   ├── ImageData.cs               # 图像数据模型（包含详细注释）
│   └── DatasetConfig.cs           # YAML配置文件模型（包含详细注释）
├── Services/                       # 服务类
│   ├── DataService.cs             # 数据加载和处理服务（包含详细注释）
│   └── TrainingService.cs         # 深度学习训练服务（包含详细注释）
├── Output/                         # 模型输出目录
│   └── pcb_detection_model.zip    # 训练好的模型文件
├── Program.cs                      # 主程序入口（包含详细注释）
├── PCBDetection.csproj            # 项目配置文件（包含详细注释）
├── PCBDetection.sln               # Visual Studio解决方案文件
└── README.md                      # 项目说明文档
```

## 缺陷类型

项目可以检测以下6种PCB缺陷（支持中英文显示）：

1. **Dry_joint - 虚焊**：焊点不牢固，接触不良
2. **Incorrect_installation - 安装错误**：元件安装位置或方向错误
3. **Short_circuit - 短路**：电路中出现意外的低阻抗连接
4. **low_solder - 少锡**：焊锡量不足，可能导致接触不良
5. **oppostie_direction - 方向错误**：有极性元件安装方向颠倒
6. **redundant - 多余元件**：不应该存在的额外元件

## 技术架构

### 深度学习框架
- **ML.NET 3.0**: 微软的机器学习框架
- **TensorFlow集成**: 底层使用TensorFlow进行深度学习计算
- **ImageClassification训练器**: 专门用于图像分类任务

### 模型架构
- **ResNet-V2-101**: 101层的残差网络架构
- **预训练模型**: 使用ImageNet预训练权重
- **迁移学习**: 在PCB数据集上进行微调

## 环境要求

### 系统要求
- **操作系统**: Windows 10/11 (推荐)
- **内存**: 至少 8GB RAM（推荐16GB）
- **存储**: 至少 5GB 可用空间
- **GPU**: 可选，支持CUDA的GPU可加速训练

### 软件要求
- **.NET 8.0 SDK**: 最新版本
- **Visual Studio 2022**: 或 Visual Studio Code
- **Git**: 用于版本控制

## 安装和运行

### 1. 克隆项目
```bash
git clone https://github.com/msumshk/PCB-Detection-MLNet.git
cd PCBDetection
```

### 2. 还原NuGet包
```bash
dotnet restore
```

### 3. 验证环境
```bash
dotnet build
```

### 4. 运行项目
```bash
dotnet run
```

## 使用方法

### 主菜单功能

运行程序后，您将看到以下菜单选项：

#### 1. 训练新模型
- 自动加载训练数据（1660张图片）和验证数据（397张图片）
- 使用ResNet-V2-101架构进行深度学习训练
- 实时显示训练进度和性能指标
- 训练完成后自动保存模型到Output目录

#### 2. 加载已有模型并测试
- 从Output目录加载之前训练好的模型
- 使用测试集（238张图片）评估模型性能
- 显示详细的评估指标和混淆矩阵

#### 3. 预测单张图片
- 输入图片文件路径
- 使用训练好的模型进行缺陷检测
- 显示预测的缺陷类型（中英文格式）

#### 4. 退出程序

## 配置文件说明

### data.yaml 配置文件
```yaml
# 数据集路径配置
train: ./train/images      # 训练图片目录路径
val: ./valid/images        # 验证图片目录路径  
test: ./test/images        # 测试图片目录路径

# 类别配置
nc: 6                      # 缺陷类别总数
names: ['Dry_joint', 'Incorrect_installation', 'Short_circuit', 'low_solder', 'oppostie_direction', 'redundant']
```

## 联系方式

如有问题或建议，请通过以下方式联系：

- **作者**: MsuMshk
- **邮箱**: heaize404@163.com
- **GitHub**: https://github.com/msumshk/PCB-Detection-MLNet
- 提交 GitHub Issue
- 参与项目讨论
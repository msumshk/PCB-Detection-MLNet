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

## 代码注释说明

本项目的所有代码文件都包含详细的中文注释：

### 1. 文档注释（XML Documentation）
- 每个类、方法、属性都有完整的 `<summary>` 注释
- 参数说明使用 `<param>` 标签
- 返回值说明使用 `<returns>` 标签
- 异常说明使用 `<exception>` 标签

### 2. 行内注释
- 关键代码行都有详细的中文解释
- 复杂逻辑有分步骤说明
- 重要配置参数有用途说明

### 3. 代码结构注释
- 每个代码段都有功能说明
- 数据流向有清晰标注
- 算法步骤有详细描述

## 缺陷类型

项目可以检测以下6种PCB缺陷（支持中英文显示）：

1. **Dry_joint - 虚焊**：焊点不牢固，接触不良
2. **Incorrect_installation - 安装错误**：元件安装位置或方向错误
3. **Short_circuit - 短路**：电路中出现意外的低阻抗连接
4. **low_solder - 少锡**：焊锡量不足，可能导致接触不良
5. **oppostie_direction - 方向错误**：有极性元件安装方向颠倒
6. **redundant - 多余元件**：不应该存在的额外元件

### 预测结果格式
程序会以中英文对照的格式显示预测结果，例如：
- `Dry_joint - 虚焊 (置信度: 85.67%)`
- `Incorrect_installation - 安装错误 (置信度: 92.34%)`

## 技术架构

### 深度学习框架
- **ML.NET 3.0**: 微软的机器学习框架
- **TensorFlow集成**: 底层使用TensorFlow进行深度学习计算
- **ImageClassification训练器**: 专门用于图像分类任务

### 模型架构
- **ResNet-V2-101**: 101层的残差网络架构
- **预训练模型**: 使用ImageNet预训练权重
- **迁移学习**: 在PCB数据集上进行微调

### 数据处理
- **YOLO格式支持**: 兼容标准YOLO数据集格式
- **图像预处理**: 自动调整图像尺寸和格式
- **标签转换**: 将字符串标签转换为数值键

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
git clone <repository-url>
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
```
功能说明：
- 自动加载训练数据（1660张图片）和验证数据（397张图片）
- 使用ResNet-V2-101架构进行深度学习训练
- 实时显示训练进度和性能指标
- 训练完成后自动保存模型到Output目录
- 在验证集上评估模型性能

训练过程：
1. 初始化TensorFlow环境
2. 加载和预处理数据
3. 构建深度学习管道
4. 执行模型训练
5. 评估和保存模型
```

#### 2. 加载已有模型并测试
```
功能说明：
- 从Output目录加载之前训练好的模型
- 使用测试集（238张图片）评估模型性能
- 显示详细的评估指标和混淆矩阵

评估指标：
- 宏平均准确率（Macro Accuracy）
- 微平均准确率（Micro Accuracy）
- 对数损失（Log Loss）
- 混淆矩阵（Confusion Matrix）
- 每个类别的详细指标
```

#### 3. 预测单张图片
```
功能说明：
- 输入图片文件路径
- 使用训练好的模型进行缺陷检测
- 显示预测的缺陷类型

支持格式：
- JPG、JPEG、PNG格式的图片文件
- 任意尺寸（程序会自动调整）
```

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

### 项目配置文件（PCBDetection.csproj）
```xml
<!-- 包含详细的中文注释，说明每个NuGet包的用途 -->
- Microsoft.ML 3.0.1: ML.NET核心包
- Microsoft.ML.Vision 3.0.1: 图像分类功能
- Microsoft.ML.ImageAnalytics 3.0.1: 图像预处理
- Microsoft.ML.TensorFlow 3.0.1: TensorFlow集成
- SciSharp.TensorFlow.Redist 2.3.1: TensorFlow运行时
- YamlDotNet 15.1.2: YAML文件解析
```

## 训练参数详解

### ImageClassification训练器配置
```csharp
var options = new ImageClassificationTrainer.Options()
{
    FeatureColumnName = "Image",        // 图像特征列名
    LabelColumnName = "LabelAsKey",     // 标签列名
    ValidationSet = preprocessedValidation, // 验证数据集
    Arch = ImageClassificationTrainer.Architecture.ResnetV2101, // ResNet架构
    TestOnTrainSet = false              // 不在训练集上测试
};
```

### 数据预处理管道
```csharp
var pipeline = labelKeyPipeline
    .Append(imagePipeline)              // 图像加载
    .Append(imageClassificationTrainer) // 深度学习训练器
    .Append(keyToValueMapper);          // 结果转换
```

## 性能指标说明

### 主要评估指标

1. **宏平均准确率（Macro Accuracy）**
   - 计算每个类别准确率的平均值
   - 对类别不平衡敏感

2. **微平均准确率（Micro Accuracy）**
   - 基于总体正确预测数量计算
   - 受样本数量多的类别影响较大

3. **对数损失（Log Loss）**
   - 衡量预测概率与真实标签的差异
   - 值越小表示模型性能越好

4. **混淆矩阵（Confusion Matrix）**
   - 显示每个类别的预测情况
   - 帮助识别容易混淆的类别

### 每类别指标
- 每个缺陷类型的详细性能指标
- 帮助识别模型在特定缺陷类型上的表现

## 代码架构说明

### 1. Program.cs - 主程序
```csharp
/// <summary>
/// PCB缺陷检测主程序类
/// 使用ML.NET深度学习框架进行PCB电路板缺陷检测
/// 支持模型训练、测试和单张图片预测功能
/// </summary>
```
- 程序入口点和用户界面
- 菜单系统和用户交互
- 异常处理和错误提示

### 2. Models/ - 数据模型
```csharp
/// <summary>
/// 图像数据模型类 - 表示训练和预测时的图像数据结构
/// 数据集配置类 - 从YAML配置文件中读取数据集信息
/// </summary>
```
- ImageData: 图像路径和标签
- ImagePrediction: 预测结果
- DatasetConfig: YAML配置映射

### 3. Services/ - 服务层
```csharp
/// <summary>
/// 数据服务类 - 负责加载和处理PCB缺陷检测的训练数据
/// 训练服务类 - 负责深度学习模型的训练、评估和预测
/// </summary>
```
- DataService: 数据加载和预处理
- TrainingService: 模型训练和评估

## 故障排除

### 常见问题及解决方案

#### 1. 内存不足错误
```
问题：训练过程中出现OutOfMemoryException
解决方案：
- 关闭其他占用内存的应用程序
- 减少训练数据量
- 使用更小的模型架构
```

#### 2. 找不到数据文件
```
问题：程序提示找不到data.yaml或图片文件
解决方案：
- 检查PCB_detect_6_700_yolo目录是否存在
- 验证data.yaml中的路径配置
- 确保图片和标签文件完整
```

#### 3. TensorFlow初始化失败
```
问题：TensorFlow环境初始化错误
解决方案：
- 确保安装了正确版本的TensorFlow运行时
- 检查系统是否支持所需的TensorFlow版本
- 尝试重新安装NuGet包
```

#### 4. 训练速度慢
```
问题：模型训练时间过长
解决方案：
- 使用支持CUDA的GPU
- 减少训练轮数
- 使用更小的图片分辨率
- 减少训练数据量
```

## 扩展功能

### 1. 添加新的缺陷类型
```yaml
# 修改data.yaml文件
nc: 7  # 增加类别数量
names: ['Dry_joint', 'Incorrect_installation', 'Short_circuit', 'low_solder', 'oppostie_direction', 'redundant', 'new_defect']
```

### 2. 调整训练参数
```csharp
// 在TrainingService.cs中修改训练配置
var options = new ImageClassificationTrainer.Options()
{
    // 调整架构
    Arch = ImageClassificationTrainer.Architecture.ResnetV250,
    // 其他参数...
};
```

### 3. 添加数据增强
```csharp
// 在数据预处理阶段添加图像变换
var augmentationPipeline = mlContext.Transforms
    .ResizeImages("ResizedImage", 224, 224, "Image")
    .Append(mlContext.Transforms.ExtractPixels("Pixels", "ResizedImage"));
```

### 4. 自定义评估指标
```csharp
// 添加自定义评估逻辑
public void CustomEvaluateModel(ITransformer model, IDataView testData)
{
    // 自定义评估代码
}
```

## 许可证

本项目采用 MIT 许可证。详情请参阅 LICENSE 文件。

## 贡献指南

欢迎提交Issue和Pull Request来改进项目：

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 GitHub Issue
- 发送邮件至项目维护者
- 参与项目讨论 
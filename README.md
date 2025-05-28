# PCB 缺陷检测 - ML.NET 深度学习项目

这是一个基于 ML.NET 深度学习框架的 PCB（印刷电路板）缺陷检测项目。该项目使用先进的深度学习技术，能够自动识别和分类 PCB 电路板上的各种缺陷类型。

## 🚀 项目特色

### 多深度学习框架支持

- **TensorFlow ImageClassification**: 原生 TensorFlow 深度学习，性能最优
- **ONNX 深度学习**: 使用 ONNX 模型进行特征提取 + ML.NET 分类器
- **智能自动选择**: 根据系统环境自动选择最适合的深度学习方法

### 跨平台兼容性

- **Windows**: 完全支持，推荐使用 TensorFlow 方法
- **macOS**: 特别优化，Apple Silicon 自动使用 ONNX 方法
- **Linux**: 完全支持，自动选择最优方法

### 智能环境检测

- 自动检测系统架构（x64/ARM64）
- 自动选择最适合的深度学习框架
- 智能降级策略，确保在任何环境下都能运行

## 📊 数据集信息

### 支持的缺陷类型

本项目能够检测以下 6 种 PCB 缺陷：

1. **Dry_joint** - 虚焊：焊点接触不良
2. **Incorrect_installation** - 安装错误：元件安装位置或方向错误
3. **Short_circuit** - 短路：电路意外连通
4. **low_solder** - 少锡：焊锡量不足
5. **oppostie_direction** - 方向错误：元件方向装反
6. **redundant** - 多余元件：不应存在的元件

### 数据集规模

- **训练集**: 1,660 张图片
- **验证集**: 397 张图片
- **测试集**: 238 张图片
- **总计**: 2,295 张高质量 PCB 图片

### 数据格式

- **图片格式**: JPG/PNG，支持多种分辨率
- **标注格式**: YOLO 格式，包含边界框和类别信息
- **目录结构**: 标准的 train/valid/test 分离

## 🛠️ 技术架构

### 深度学习框架

- **主要方法**: TensorFlow ImageClassification（ML.NET 3.0 原生支持）
- **备用方法**: ONNX 深度学习特征提取 + ML.NET 分类器
- **自动选择**: 智能检测并选择最适合当前环境的方法

### 核心技术栈

- **ML.NET 3.0**: 微软的机器学习框架
- **TensorFlow 集成**: 底层使用 TensorFlow 进行深度学习计算
- **ONNX 支持**: 跨平台深度学习模型标准
- **ImageClassification 训练器**: 专门用于图像分类任务

### 模型架构选择

- **TensorFlow 方法**:
  - Apple Silicon: MobilenetV2（轻量级，适合 ARM 架构）
  - Intel/AMD: ResnetV250（标准架构，性能优秀）
- **ONNX 方法**: ResNet18（平衡性能和兼容性）

### 数据处理

- **YOLO 格式支持**: 兼容标准 YOLO 数据集格式
- **图像预处理**: 自动调整图像尺寸和格式
- **标签转换**: 将字符串标签转换为数值键

## 🔧 环境要求

### 系统要求

- **操作系统**: Windows 10/11, macOS 10.15+, Linux
- **内存**: 至少 8GB RAM（推荐 16GB）
- **存储**: 至少 5GB 可用空间
- **GPU**: 可选，支持 CUDA 的 GPU 可加速训练（仅 Windows/Linux）

### 软件要求

- **.NET 8.0 SDK**: 最新版本
- **Visual Studio 2022**: 或 Visual Studio Code
- **Git**: 用于版本控制

## 📦 安装和运行

### 1. 克隆项目

```bash
git clone <repository-url>
cd PCBDetection
```

### 2. 还原 NuGet 包

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

## 🎯 使用方法

### 主菜单功能

运行程序后，您将看到以下菜单选项：

#### 1. 训练新模型

```
功能说明：
- 自动检测最适合的深度学习方法
- 智能选择TensorFlow或ONNX框架
- 自动加载训练数据（1660张图片）和验证数据（397张图片）
- 实时显示训练进度和性能指标
- 训练完成后自动保存模型到Output目录
- 在验证集上评估模型性能

深度学习方法选择：
1. 首先尝试TensorFlow ImageClassification（原生深度学习）
2. 如果TensorFlow不可用，自动切换到ONNX方法
3. 根据系统架构选择最优的模型架构

训练过程：
1. 检测系统环境和可用框架
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
- 显示预测的缺陷类型和置信度

支持格式：
- JPG、JPEG、PNG格式的图片文件
- 任意尺寸（程序会自动调整）
```

#### 4. 退出程序

## ⚙️ 配置文件说明

### data.yaml 配置文件

```yaml
# 数据集路径配置
train: ./train/images # 训练图片目录路径
val: ./valid/images # 验证图片目录路径
test: ./test/images # 测试图片目录路径

# 类别配置
nc: 6 # 缺陷类别总数
names:
  [
    "Dry_joint",
    "Incorrect_installation",
    "Short_circuit",
    "low_solder",
    "oppostie_direction",
    "redundant",
  ]
```

### 项目配置文件（PCBDetection.csproj）

```xml
<!-- 深度学习支持包 - 多种选择 -->
<!-- TensorFlow支持（主要选择） -->
- Microsoft.ML.TensorFlow 3.0.1: TensorFlow集成
- SciSharp.TensorFlow.Redist 2.16.0: TensorFlow运行时

<!-- ONNX支持（备用选择） -->
- Microsoft.ML.OnnxRuntime 1.16.3: ONNX运行时
- Microsoft.ML.OnnxTransformer 3.0.1: ONNX转换器

<!-- 其他核心包 -->
- Microsoft.ML 3.0.1: ML.NET核心包
- Microsoft.ML.Vision 3.0.1: 图像分类功能
- Microsoft.ML.ImageAnalytics 3.0.1: 图像预处理
- YamlDotNet 15.1.2: YAML文件解析
```

## 🔬 深度学习方法详解

### 方法 1: TensorFlow ImageClassification（推荐）

```csharp
// 原生TensorFlow深度学习
var options = new ImageClassificationTrainer.Options()
{
    FeatureColumnName = "Image",
    LabelColumnName = "LabelAsKey",
    ValidationSet = preprocessedValidation,
    Arch = architecture,  // 根据系统自动选择
    TestOnTrainSet = false,
    Epoch = isAppleSilicon ? 10 : 15,
    BatchSize = isAppleSilicon ? 8 : 16,
    LearningRate = 0.01f
};
```

**优势:**

- 端到端深度学习训练
- 性能最优，准确率最高
- 支持多种预训练架构
- GPU 加速支持（Windows/Linux）

**适用环境:**

- Windows（推荐）
- Linux（推荐）
- macOS Intel（可用）

### 方法 2: ONNX 深度学习特征提取

```csharp
// ONNX特征提取 + ML.NET分类器
var pipeline = mlContext.Transforms.DnnFeaturizeImage("Features",
    modelFactory => modelFactory.ResNet18(mlContext, "Features", "ImagePixels"),
    "ImagePixels")
.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(...));
```

**优势:**

- 跨平台兼容性最佳
- Apple Silicon 原生支持
- 稳定可靠
- 无需 GPU 依赖

**适用环境:**

- macOS Apple Silicon（推荐）
- 任何 TensorFlow 不可用的环境

### 智能自动选择逻辑

```
1. 检测系统环境（操作系统、架构）
2. 尝试初始化TensorFlow ImageClassification
3. 如果成功 → 使用TensorFlow方法
4. 如果失败 → 自动切换到ONNX方法
5. 根据系统架构选择最优模型架构
```

## 📈 性能指标说明

### 主要评估指标

1. **宏平均准确率 (Macro Accuracy)**

   - 计算每个类别准确率的平均值
   - 适用于类别不平衡的数据集
   - 范围: 0.0 - 1.0，越高越好

2. **微平均准确率 (Micro Accuracy)**

   - 基于所有样本计算的整体准确率
   - 更关注样本数量多的类别
   - 范围: 0.0 - 1.0，越高越好

3. **对数损失 (Log Loss)**

   - 衡量预测概率与真实标签的差异
   - 范围: 0.0 - ∞，越低越好
   - 值接近 0 表示模型预测非常准确

4. **混淆矩阵 (Confusion Matrix)**
   - 显示每个类别的预测结果分布
   - 对角线元素表示正确预测
   - 非对角线元素表示错误分类

### 性能优化建议

#### 提高准确率

- 增加训练数据量
- 使用数据增强技术
- 调整学习率和训练轮数
- 尝试不同的模型架构

#### 减少训练时间

- 使用 GPU 加速（如果可用）
- 减少 Epoch 数量
- 使用较小的 BatchSize
- 选择轻量级模型架构

#### 平台特定优化

- **Apple Silicon**: 自动使用 ONNX 方法，优化内存使用
- **Intel/AMD**: 优先使用 TensorFlow，支持更大的 BatchSize
- **GPU 环境**: 自动启用 GPU 加速（Windows/Linux）

## 🔍 故障排除

### 常见问题

#### 1. TensorFlow 初始化失败

```
症状: "TensorFlow exception triggered while loading model"
解决方案:
- 程序会自动切换到ONNX方法
- 手动指定使用ONNX: TrainModel(DeepLearningMethod.OnnxFeaturizer)
- 检查.NET 8.0 SDK版本
```

#### 2. 内存不足

```
症状: OutOfMemoryException
解决方案:
- 减少BatchSize（Apple Silicon默认已优化）
- 关闭其他应用程序
- 使用轻量级模型架构
```

#### 3. Apple Silicon 兼容性

```
症状: 在M1/M2 Mac上运行缓慢或失败
解决方案:
- 程序会自动选择ONNX方法
- 确保安装了ARM64版本的.NET SDK
- 设置环境变量: DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
```

#### 4. 数据加载失败

```
症状: "数据集目录不存在"
解决方案:
- 检查PCB_detect_6_700_yolo目录是否存在
- 验证data.yaml配置文件路径
- 确保图片和标签文件匹配
```

## 🚀 高级功能

### 自定义深度学习方法

```csharp
// 强制使用特定方法
var model = trainingService.TrainModel(DeepLearningMethod.TensorFlowImageClassification);
// 或
var model = trainingService.TrainModel(DeepLearningMethod.OnnxFeaturizer);
```

### 模型架构选择

程序会根据系统自动选择最优架构：

- **Apple Silicon**: MobilenetV2（轻量级，ARM 优化）
- **Intel/AMD**: ResnetV250（标准性能）
- **ONNX 方法**: ResNet18（平衡性能和兼容性）

### 训练参数优化

系统会根据硬件自动调整：

- **Epoch**: Apple Silicon 使用较少轮数避免过热
- **BatchSize**: 根据内存容量自动调整
- **LearningRate**: 统一使用 0.01f 获得稳定训练

## 📝 开发说明

### 项目结构

```
PCBDetection/
├── Models/                 # 数据模型定义
│   ├── ImageData.cs       # 图像数据模型
│   └── DatasetConfig.cs   # 数据集配置模型
├── Services/              # 业务逻辑服务
│   ├── TrainingService.cs # 训练服务（多框架支持）
│   └── DataService.cs     # 数据加载服务
├── PCB_detect_6_700_yolo/ # 数据集目录
├── Output/                # 模型输出目录
├── data.yaml             # 数据集配置文件
└── Program.cs            # 主程序入口
```

### 扩展开发

- 添加新的深度学习方法
- 支持更多模型架构
- 集成其他深度学习框架
- 添加模型量化和优化

## 📄 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 这是一个教育和研究项目，用于演示 ML.NET 深度学习在工业缺陷检测中的应用。在生产环境中使用前，请进行充分的测试和验证。

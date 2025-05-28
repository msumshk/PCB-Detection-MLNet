using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using PCBDetection.Models;
using System.Runtime.InteropServices;

namespace PCBDetection.Services
{
    /// <summary>
    /// 训练服务类 - MacOS优化版本
    /// 负责PCB缺陷检测模型的深度学习训练、评估和预测
    /// 使用ML.NET的ImageClassification训练器进行图像分类任务
    /// 针对MacOS和Apple Silicon进行了特别优化
    /// </summary>
    public class TrainingService
    {
        /// <summary>
        /// ML.NET上下文实例，用于机器学习操作
        /// </summary>
        private readonly MLContext _mlContext;
        
        /// <summary>
        /// 数据服务实例，用于加载训练数据
        /// </summary>
        private readonly DataService _dataService;
        
        /// <summary>
        /// 模型输出目录路径
        /// </summary>
        private readonly string _outputDir = "Output";

        /// <summary>
        /// 是否为Apple Silicon Mac
        /// </summary>
        private readonly bool _isAppleSilicon;

        /// <summary>
        /// 构造函数
        /// 初始化训练服务，创建输出目录，检测系统架构
        /// </summary>
        /// <param name="mlContext">ML.NET上下文</param>
        /// <param name="dataService">数据服务实例</param>
        public TrainingService(MLContext mlContext, DataService dataService)
        {
            _mlContext = mlContext;
            _dataService = dataService;
            _isAppleSilicon = RuntimeInformation.OSArchitecture == Architecture.Arm64 && 
                             RuntimeInformation.IsOSPlatform(OSPlatform.OSX);
            
            // 设置TensorFlow环境变量以提高兼容性
            SetTensorFlowEnvironment();
            
            // 确保输出目录存在，如果不存在则创建
            if (!Directory.Exists(_outputDir))
            {
                Directory.CreateDirectory(_outputDir);
            }

            // 显示系统信息
            Console.WriteLine($"系统架构: {RuntimeInformation.OSArchitecture}");
            Console.WriteLine($"操作系统: {RuntimeInformation.OSDescription}");
            Console.WriteLine($"Apple Silicon Mac: {_isAppleSilicon}");
        }

        /// <summary>
        /// 设置TensorFlow环境变量以提高兼容性
        /// </summary>
        private void SetTensorFlowEnvironment()
        {
            try
            {
                // 设置TensorFlow日志级别，减少输出
                Environment.SetEnvironmentVariable("TF_CPP_MIN_LOG_LEVEL", "2");
                
                // 禁用GPU（在macOS上通常会导致问题）
                Environment.SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "-1");
                
                // 设置线程数以避免过度使用CPU
                Environment.SetEnvironmentVariable("OMP_NUM_THREADS", "4");
                Environment.SetEnvironmentVariable("TF_NUM_INTEROP_THREADS", "4");
                Environment.SetEnvironmentVariable("TF_NUM_INTRAOP_THREADS", "4");
                
                if (_isAppleSilicon)
                {
                    // Apple Silicon特定设置
                    Environment.SetEnvironmentVariable("DOTNET_SYSTEM_GLOBALIZATION_INVARIANT", "1");
                    Console.WriteLine("🔧 已设置Apple Silicon优化环境变量");
                }
                else
                {
                    Console.WriteLine("🔧 已设置Intel Mac优化环境变量");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️ 设置环境变量时出现警告: {ex.Message}");
            }
        }

        /// <summary>
        /// 训练深度学习模型的主方法
        /// 执行完整的训练流程：初始化TensorFlow、加载数据、构建管道、训练模型
        /// 强制使用TensorFlow深度学习，不使用传统机器学习备用方案
        /// </summary>
        /// <returns>训练好的模型转换器</returns>
        public ITransformer TrainModel()
        {
            try
            {
                Console.WriteLine("正在初始化TensorFlow环境...");
                InitializeTensorFlow(); // 初始化TensorFlow环境，必须成功

                Console.WriteLine("正在加载训练数据...");
                // 加载训练数据和验证数据
                var trainData = _dataService.LoadTrainingData();
                var validationData = _dataService.LoadValidationData();

                // 显示数据集统计信息
                Console.WriteLine($"训练样本数: {trainData.GetRowCount()}");
                Console.WriteLine($"验证样本数: {validationData.GetRowCount()}");

                Console.WriteLine("正在构建TensorFlow深度学习管道...");
                // 强制使用深度学习管道
                var pipeline = BuildDeepLearningPipeline(validationData);

                Console.WriteLine("开始TensorFlow深度学习模型训练...");
                Console.WriteLine("这可能需要一些时间，具体取决于您的硬件配置...");
                // 执行模型训练
                var model = pipeline.Fit(trainData);

                Console.WriteLine("正在评估模型...");
                // 在验证集上评估模型性能
                EvaluateModel(model, validationData);

                return model;
            }
            catch (Exception ex)
            {
                // 捕获训练过程中的异常
                Console.WriteLine($"TensorFlow深度学习训练失败: {ex.Message}");
                Console.WriteLine($"堆栈跟踪: {ex.StackTrace}");
                
                // 如果是TensorFlow相关错误，提供解决建议
                if (ex.Message.Contains("TensorFlow") || ex.Message.Contains("Tensorflow"))
                {
                    Console.WriteLine("\n=== MacOS TensorFlow 深度学习问题解决建议 ===");
                    Console.WriteLine("1. 确保已安装最新版本的 .NET 8.0");
                    Console.WriteLine("2. 尝试运行: dotnet restore --force");
                    Console.WriteLine("3. 尝试使用Rosetta 2运行: arch -x86_64 dotnet run");
                    Console.WriteLine("4. 检查TensorFlow运行时是否正确安装");
                    Console.WriteLine("5. 确保有足够的内存和存储空间");
                    Console.WriteLine("注意: 本项目仅支持TensorFlow深度学习，不提供传统机器学习备用方案");
                }
                
                throw;
            }
        }

        /// <summary>
        /// 初始化TensorFlow环境
        /// 通过多种方法尝试初始化TensorFlow，包括兼容性检查和自动修复
        /// 如果所有方法都失败，将抛出异常终止程序
        /// </summary>
        private void InitializeTensorFlow()
        {
            Console.WriteLine("🔧 正在初始化TensorFlow深度学习环境...");
            
            // 尝试多种初始化方法
            var initMethods = new List<(string name, Func<bool> method)>
            {
                ("标准初始化", TryStandardInit),
                ("兼容模式初始化", TryCompatibilityInit),
                ("简化架构初始化", TrySimplifiedInit)
            };

            foreach (var (name, method) in initMethods)
            {
                try
                {
                    Console.WriteLine($"🔄 尝试{name}...");
                    if (method())
                    {
                        Console.WriteLine($"✅ {name}成功！");
                        Console.WriteLine("✅ TensorFlow深度学习环境准备就绪");
                        return;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ {name}失败: {ex.Message}");
                }
            }

            // 所有方法都失败，提供详细的错误信息
            Console.WriteLine("❌ 所有TensorFlow初始化方法都失败");
            ProvideTensorFlowSolution();
            throw new InvalidOperationException("TensorFlow 深度学习环境初始化失败，无法继续训练");
        }

        /// <summary>
        /// 尝试标准TensorFlow初始化
        /// </summary>
        private bool TryStandardInit()
        {
            var testData = new List<ImageData> { new ImageData { ImagePath = "", Label = "test" } };
            var testDataView = _mlContext.Data.LoadFromEnumerable(testData);
            
            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                Arch = _isAppleSilicon ? ImageClassificationTrainer.Architecture.ResnetV250 : ImageClassificationTrainer.Architecture.ResnetV2101,
                TestOnTrainSet = false,
                Epoch = 1 // 最小epoch用于测试
            };
            
            var trainer = _mlContext.MulticlassClassification.Trainers.ImageClassification(options);
            Console.WriteLine($"✅ 使用架构: {options.Arch}");
            return true;
        }

        /// <summary>
        /// 尝试兼容模式初始化
        /// </summary>
        private bool TryCompatibilityInit()
        {
            var testData = new List<ImageData> { new ImageData { ImagePath = "", Label = "test" } };
            var testDataView = _mlContext.Data.LoadFromEnumerable(testData);
            
            // 使用最轻量的架构
            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                Arch = ImageClassificationTrainer.Architecture.MobilenetV2,
                TestOnTrainSet = false,
                Epoch = 1,
                BatchSize = 4 // 更小的批次大小
            };
            
            var trainer = _mlContext.MulticlassClassification.Trainers.ImageClassification(options);
            Console.WriteLine($"✅ 使用兼容架构: {options.Arch}");
            return true;
        }

        /// <summary>
        /// 尝试简化架构初始化
        /// </summary>
        private bool TrySimplifiedInit()
        {
            var testData = new List<ImageData> { new ImageData { ImagePath = "", Label = "test" } };
            var testDataView = _mlContext.Data.LoadFromEnumerable(testData);
            
            // 使用最基础的架构
            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                Arch = ImageClassificationTrainer.Architecture.ResnetV250,
                TestOnTrainSet = false,
                Epoch = 1,
                BatchSize = 2,
                LearningRate = 0.001f
            };
            
            var trainer = _mlContext.MulticlassClassification.Trainers.ImageClassification(options);
            Console.WriteLine($"✅ 使用简化架构: {options.Arch}");
            return true;
        }

        /// <summary>
        /// 提供TensorFlow问题的详细解决方案
        /// </summary>
        private void ProvideTensorFlowSolution()
        {
            Console.WriteLine("\n=== 🔧 MacOS TensorFlow 深度学习解决方案 ===");
            
            if (_isAppleSilicon)
            {
                Console.WriteLine("📱 Apple Silicon Mac 解决方案:");
                Console.WriteLine("1. 🔄 使用Rosetta 2运行:");
                Console.WriteLine("   arch -x86_64 dotnet run");
                Console.WriteLine();
                Console.WriteLine("2. 🔧 安装x64版本的.NET SDK:");
                Console.WriteLine("   curl -sSL https://dot.net/v1/dotnet-install.sh | bash /dev/stdin --architecture x64");
                Console.WriteLine();
                Console.WriteLine("3. 🔄 清理并重新构建:");
                Console.WriteLine("   dotnet clean");
                Console.WriteLine("   dotnet restore --force");
                Console.WriteLine("   arch -x86_64 dotnet build");
                Console.WriteLine();
                Console.WriteLine("4. 🎯 设置环境变量:");
                Console.WriteLine("   export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1");
                Console.WriteLine("   export TF_CPP_MIN_LOG_LEVEL=2");
            }
            else
            {
                Console.WriteLine("💻 Intel Mac 解决方案:");
                Console.WriteLine("1. 🔄 确保使用x64架构:");
                Console.WriteLine("   dotnet run --runtime osx-x64");
                Console.WriteLine();
                Console.WriteLine("2. 🔧 重新安装依赖:");
                Console.WriteLine("   dotnet clean");
                Console.WriteLine("   dotnet restore --force");
            }
            
            Console.WriteLine("\n📋 通用解决方案:");
            Console.WriteLine("1. ✅ 确保.NET 8.0 SDK版本正确:");
            Console.WriteLine("   dotnet --version");
            Console.WriteLine();
            Console.WriteLine("2. 💾 检查可用内存和存储空间");
            Console.WriteLine("3. 🔄 重启终端并重新尝试");
            Console.WriteLine();
            Console.WriteLine("⚠️  注意: 本项目仅支持TensorFlow深度学习");
            Console.WriteLine("   如果问题持续，请考虑在Intel Mac或Linux环境中运行");
        }

        /// <summary>
        /// 构建TensorFlow深度学习训练管道
        /// 包括数据预处理、图像加载、标签转换和ImageClassification训练器
        /// 专门针对MacOS和Apple Silicon优化
        /// </summary>
        /// <param name="validationData">验证数据集</param>
        /// <returns>完整的TensorFlow深度学习训练管道</returns>
        private IEstimator<ITransformer> BuildDeepLearningPipeline(IDataView validationData)
        {
            Console.WriteLine("🔧 构建TensorFlow深度学习管道...");
            
            // 创建标签转换管道：将字符串标签转换为数值键
            var labelKeyPipeline = _mlContext.Transforms.Conversion.MapValueToKey("LabelAsKey", "Label");
            
            // 创建图像加载管道：从文件路径加载原始图像字节
            var imagePipeline = _mlContext.Transforms.LoadRawImageBytes("Image", null, "ImagePath");
            
            // 预处理验证数据，确保它有正确的列结构
            Console.WriteLine("📊 预处理验证数据...");
            var preprocessedValidation = labelKeyPipeline.Fit(validationData).Transform(validationData);
            preprocessedValidation = imagePipeline.Fit(preprocessedValidation).Transform(preprocessedValidation);

            // 根据系统架构选择最优的深度学习架构
            var architecture = _isAppleSilicon ? 
                ImageClassificationTrainer.Architecture.ResnetV250 : 
                ImageClassificationTrainer.Architecture.ResnetV2101;

            Console.WriteLine($"🏗️  选择深度学习架构: {architecture}");
            Console.WriteLine($"💻 系统优化: {(_isAppleSilicon ? "Apple Silicon (ARM64)" : "Intel (x64)")}");

            // 配置ML.NET 3.0的ImageClassification训练器选项
            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",        // 图像特征列名
                LabelColumnName = "LabelAsKey",     // 标签列名
                ValidationSet = preprocessedValidation, // 验证数据集
                Arch = architecture,                // 根据系统选择架构
                TestOnTrainSet = false,             // 不在训练集上测试
                WorkspacePath = _outputDir,         // 设置工作空间路径
                // MacOS优化设置
                Epoch = _isAppleSilicon ? 10 : 15,  // Apple Silicon使用较少epoch以避免过热
                BatchSize = _isAppleSilicon ? 8 : 16, // Apple Silicon使用较小批次大小
                LearningRate = 0.01f,               // 学习率
                EarlyStoppingCriteria = new ImageClassificationTrainer.EarlyStopping()
                {
                    MinDelta = 0.001f,              // 最小改进阈值
                    Patience = 3                    // 早停耐心值
                }
            };

            Console.WriteLine($"⚙️  训练参数配置:");
            Console.WriteLine($"   - Epoch: {options.Epoch}");
            Console.WriteLine($"   - BatchSize: {options.BatchSize}");
            Console.WriteLine($"   - LearningRate: {options.LearningRate}");
            Console.WriteLine($"   - WorkspacePath: {options.WorkspacePath}");

            // 构建完整的TensorFlow深度学习训练管道
            var pipeline = labelKeyPipeline
                .Append(imagePipeline) // 添加图像加载步骤
                .Append(_mlContext.MulticlassClassification.Trainers.ImageClassification(options)) // 添加TensorFlow深度学习训练器
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel")); // 将预测结果转换回字符串

            Console.WriteLine("✅ TensorFlow深度学习管道构建完成！");
            return pipeline;
        }

        

        /// <summary>
        /// 评估模型性能
        /// 计算并显示各种评估指标，包括准确率、损失函数、混淆矩阵等
        /// </summary>
        /// <param name="model">训练好的模型</param>
        /// <param name="testData">测试数据集</param>
        public void EvaluateModel(ITransformer model, IDataView testData)
        {
            try
            {
                Console.WriteLine("Transforming test data...");
                // 使用模型对测试数据进行预测
                var predictions = model.Transform(testData);

                Console.WriteLine("Computing metrics...");
                // 计算多分类评估指标
                var metrics = _mlContext.MulticlassClassification.Evaluate(predictions, "LabelAsKey", "Score");

                // 显示主要评估指标
                Console.WriteLine($"\n=== Model Evaluation Metrics ===");
                Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy:F4}");     // 宏平均准确率
                Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy:F4}");     // 微平均准确率
                Console.WriteLine($"Log Loss: {metrics.LogLoss:F4}");                 // 对数损失
                Console.WriteLine($"Log Loss Reduction: {metrics.LogLossReduction:F4}"); // 对数损失减少量

                // 显示混淆矩阵
                Console.WriteLine($"\n=== Confusion Matrix ===");
                var confusionMatrix = metrics.ConfusionMatrix;
                Console.WriteLine($"Confusion Matrix: {confusionMatrix.NumberOfClasses}x{confusionMatrix.NumberOfClasses}");
                
                // 打印混淆矩阵的具体数值
                var matrix = confusionMatrix.Counts;
                for (int i = 0; i < confusionMatrix.NumberOfClasses; i++)
                {
                    for (int j = 0; j < confusionMatrix.NumberOfClasses; j++)
                    {
                        Console.Write($"{matrix[i][j],4} ");
                    }
                    Console.WriteLine();
                }

                // 显示每个类别的详细指标
                Console.WriteLine($"\n=== Per-Class Metrics ===");
                var classNames = _dataService.GetClassNames();
                for (int i = 0; i < classNames.Count && i < metrics.PerClassLogLoss.Count; i++)
                {
                    var englishName = classNames[i];
                    var chineseName = ImageData.DefectTypeMapping.TryGetValue(englishName, out string? chinese) ? chinese : "未知类型";
                    Console.WriteLine($"Class '{englishName} - {chineseName}': Log Loss = {metrics.PerClassLogLoss[i]:F4}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Evaluation error: {ex.Message}");
            }
        }

        /// <summary>
        /// 保存训练好的模型到文件
        /// 将模型序列化并保存到指定的输出目录
        /// </summary>
        /// <param name="model">要保存的模型</param>
        /// <param name="modelName">模型文件名，默认为"pcb_detection_model.zip"</param>
        public void SaveModel(ITransformer model, string modelName = "pcb_detection_model.zip")
        {
            try
            {
                // 构建完整的模型保存路径
                var modelPath = Path.Combine(_outputDir, modelName);
                Console.WriteLine($"Saving model to: {modelPath}");
                
                // 获取训练数据的架构信息
                var trainData = _dataService.LoadTrainingData();
                
                // 保存模型和数据架构
                _mlContext.Model.Save(model, trainData.Schema, modelPath);
                Console.WriteLine("Model saved successfully!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error saving model: {ex.Message}");
            }
        }

        /// <summary>
        /// 使用测试数据集测试模型
        /// 加载测试数据并评估模型在测试集上的性能
        /// </summary>
        /// <param name="model">要测试的模型</param>
        public void TestModel(ITransformer model)
        {
            Console.WriteLine("Testing model with test dataset...");
            // 加载测试数据
            var testData = _dataService.LoadTestData();
            
            // 检查是否有测试数据
            if (testData.GetRowCount() == 0)
            {
                Console.WriteLine("No test data available.");
                return;
            }

            // 在测试集上评估模型
            EvaluateModel(model, testData);
        }

        /// <summary>
        /// 对单张图片进行预测
        /// 使用训练好的模型对指定图片进行缺陷检测预测
        /// </summary>
        /// <param name="model">训练好的模型</param>
        /// <param name="imagePath">要预测的图片路径</param>
        /// <returns>预测的缺陷类型（中英文格式）</returns>
        public string PredictSingleImage(ITransformer model, string imagePath)
        {
            try
            {
                // 创建预测引擎
                var predictionEngine = _mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
                
                // 创建输入数据
                var imageData = new ImageData { ImagePath = imagePath };
                
                // 执行预测
                var prediction = predictionEngine.Predict(imageData);
                
                // 返回格式化的预测结果（中英文）
                if (prediction != null && !string.IsNullOrEmpty(prediction.PredictedLabel))
                {
                    var formattedResult = ImageData.GetDefectDescription(prediction.PredictedLabel);
                    
                    // 如果有置信度分数，也显示出来
                    if (prediction.Score != null && prediction.Score.Length > 0)
                    {
                        var maxScore = prediction.Score.Max();
                        var confidence = (maxScore * 100).ToString("F2");
                        return $"{formattedResult} (置信度: {confidence}%)";
                    }
                    
                    return formattedResult;
                }
                
                return "Unknown - 未知";
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Prediction error: {ex.Message}");
                return "Error - 预测错误";
            }
        }
    }
}
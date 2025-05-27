using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using PCBDetection.Models;

namespace PCBDetection.Services
{
    /// <summary>
    /// 训练服务类
    /// 负责PCB缺陷检测模型的深度学习训练、评估和预测
    /// 使用ML.NET的ImageClassification训练器进行图像分类任务
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
        /// 构造函数
        /// 初始化训练服务，创建输出目录
        /// </summary>
        /// <param name="mlContext">ML.NET上下文</param>
        /// <param name="dataService">数据服务实例</param>
        public TrainingService(MLContext mlContext, DataService dataService)
        {
            _mlContext = mlContext;
            _dataService = dataService;
            
            // 确保输出目录存在，如果不存在则创建
            if (!Directory.Exists(_outputDir))
            {
                Directory.CreateDirectory(_outputDir);
            }
        }

        /// <summary>
        /// 训练深度学习模型的主方法
        /// 执行完整的训练流程：初始化TensorFlow、加载数据、构建管道、训练模型
        /// </summary>
        /// <returns>训练好的模型转换器</returns>
        public ITransformer TrainModel()
        {
            try
            {
                Console.WriteLine("Initializing TensorFlow environment...");
                InitializeTensorFlow(); // 初始化TensorFlow环境

                Console.WriteLine("Loading training data...");
                // 加载训练数据和验证数据
                var trainData = _dataService.LoadTrainingData();
                var validationData = _dataService.LoadValidationData();

                // 显示数据集统计信息
                Console.WriteLine($"Training samples: {trainData.GetRowCount()}");
                Console.WriteLine($"Validation samples: {validationData.GetRowCount()}");

                Console.WriteLine("Building deep learning pipeline...");
                // 构建深度学习训练管道
                var pipeline = BuildDeepLearningPipeline(validationData);

                Console.WriteLine("Starting deep learning model training...");
                Console.WriteLine("This may take a while depending on your hardware...");
                // 执行模型训练
                var model = pipeline.Fit(trainData);

                Console.WriteLine("Evaluating model...");
                // 在验证集上评估模型性能
                EvaluateModel(model, validationData);

                return model;
            }
            catch (Exception ex)
            {
                // 捕获训练过程中的异常
                Console.WriteLine($"Training failed: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                throw;
            }
        }

        /// <summary>
        /// 初始化TensorFlow环境
        /// 通过创建测试管道来验证TensorFlow是否正确安装和配置
        /// </summary>
        private void InitializeTensorFlow()
        {
            try
            {
                // 尝试初始化TensorFlow环境
                Console.WriteLine("Checking TensorFlow availability...");
                
                // 创建一个简单的测试数据来验证TensorFlow是否可用
                var testData = new List<ImageData> 
                { 
                    new ImageData { ImagePath = "", Label = "test" } 
                };
                var testDataView = _mlContext.Data.LoadFromEnumerable(testData);
                
                // 创建测试管道，这将触发TensorFlow初始化
                var testPipeline = _mlContext.Transforms.Conversion.MapValueToKey("LabelAsKey", "Label");
                var testModel = testPipeline.Fit(testDataView);
                
                Console.WriteLine("TensorFlow environment initialized successfully.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"TensorFlow initialization warning: {ex.Message}");
                // 继续执行，让实际的训练过程处理TensorFlow错误
            }
        }

        /// <summary>
        /// 构建深度学习训练管道
        /// 包括数据预处理、图像加载、标签转换和ImageClassification训练器
        /// </summary>
        /// <param name="validationData">验证数据集</param>
        /// <returns>完整的训练管道</returns>
        private IEstimator<ITransformer> BuildDeepLearningPipeline(IDataView validationData)
        {
            // 创建标签转换管道：将字符串标签转换为数值键
            var labelKeyPipeline = _mlContext.Transforms.Conversion.MapValueToKey("LabelAsKey", "Label");
            
            // 创建图像加载管道：从文件路径加载原始图像字节
            var imagePipeline = _mlContext.Transforms.LoadRawImageBytes("Image", null, "ImagePath");
            
            // 预处理验证数据，确保它有正确的列结构
            var preprocessedValidation = labelKeyPipeline.Fit(validationData).Transform(validationData);
            preprocessedValidation = imagePipeline.Fit(preprocessedValidation).Transform(preprocessedValidation);

            // 配置ML.NET 3.0的ImageClassification训练器选项
            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",        // 图像特征列名
                LabelColumnName = "LabelAsKey",     // 标签列名
                ValidationSet = preprocessedValidation, // 验证数据集
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101, // 使用ResNet V2 101架构
                TestOnTrainSet = false              // 不在训练集上测试
            };

            // 构建完整的训练管道
            var pipeline = labelKeyPipeline
                .Append(imagePipeline) // 添加图像加载步骤
                .Append(_mlContext.MulticlassClassification.Trainers.ImageClassification(options)) // 添加深度学习训练器
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel")); // 将预测结果转换回字符串

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
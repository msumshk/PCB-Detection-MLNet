using Microsoft.ML;
using PCBDetection.Services;

namespace PCBDetection
{
    /// <summary>
    /// PCB缺陷检测主程序类
    /// 使用ML.NET深度学习框架进行PCB电路板缺陷检测
    /// 支持模型训练、测试和单张图片预测功能
    /// </summary>
    class Program
    {
        /// <summary>
        /// 程序主入口点
        /// 初始化ML.NET环境，加载配置文件，提供用户交互界面
        /// </summary>
        /// <param name="args">命令行参数</param>
        static void Main(string[] args)
        {
            // 显示程序标题
            Console.WriteLine("=== PCB缺陷检测 - ML.NET深度学习训练程序 ===\n");

            // 初始化ML.NET上下文，设置随机种子为0以确保结果可重现
            var mlContext = new MLContext(seed: 0);

            try
            {
                // 查找配置文件路径 - 支持多种可能的路径位置
                string configPath = FindConfigFile();
                
                // 检查配置文件是否存在
                if (string.IsNullOrEmpty(configPath))
                {
                    Console.WriteLine("错误: 找不到配置文件 data.yaml");
                    Console.WriteLine("请确保 PCB_detect_6_700_yolo/data.yaml 文件存在");
                    Console.WriteLine($"当前工作目录: {Directory.GetCurrentDirectory()}");
                    return;
                }

                Console.WriteLine($"使用配置文件: {configPath}");

                // 初始化数据服务和训练服务
                var dataService = new DataService(mlContext, configPath);
                var trainingService = new TrainingService(mlContext, dataService);

                // 显示数据集基本信息（中英文格式）
                dataService.DisplayDatasetInfo();

                // 主程序循环 - 显示菜单并处理用户选择
                while (true)
                {
                    Console.WriteLine("\n请选择操作:");
                    Console.WriteLine("1. 训练新模型");
                    Console.WriteLine("2. 加载已有模型并测试");
                    Console.WriteLine("3. 预测单张图片");
                    Console.WriteLine("4. 退出");
                    Console.Write("请输入选择 (1-4): ");

                    var choice = Console.ReadLine();

                    // 根据用户选择执行相应操作
                    switch (choice)
                    {
                        case "1":
                            TrainNewModel(trainingService);
                            break;
                        case "2":
                            TestExistingModel(mlContext, trainingService);
                            break;
                        case "3":
                            PredictSingleImage(mlContext, trainingService);
                            break;
                        case "4":
                            Console.WriteLine("程序退出。");
                            return;
                        default:
                            Console.WriteLine("无效选择，请重新输入。");
                            break;
                    }
                }
            }
            catch (Exception ex)
            {
                // 捕获并显示程序执行过程中的异常
                Console.WriteLine($"程序执行出错: {ex.Message}");
                Console.WriteLine($"详细错误: {ex}");
            }
        }

        /// <summary>
        /// 查找配置文件的方法
        /// 在多个可能的路径中搜索data.yaml配置文件
        /// 这样可以处理从不同目录运行程序的情况
        /// </summary>
        /// <returns>配置文件的完整路径，如果找不到则返回空字符串</returns>
        static string FindConfigFile()
        {
            // 定义可能的配置文件路径数组
            // 包括当前目录和从编译输出目录回溯到项目根目录的路径
            var possiblePaths = new[]
            {
                Path.Combine("PCB_detect_6_700_yolo", "data.yaml"),                    // 当前目录下的数据集文件夹
                Path.Combine("..", "..", "..", "PCB_detect_6_700_yolo", "data.yaml"), // 从bin/Debug/net8.0回到项目根目录
                Path.Combine("..", "..", "..", "..", "PCB_detect_6_700_yolo", "data.yaml"), // 额外的备用路径
            };

            // 遍历所有可能的路径，返回第一个存在的文件路径
            foreach (var path in possiblePaths)
            {
                if (File.Exists(path))
                {
                    return Path.GetFullPath(path); // 返回绝对路径
                }
            }

            return string.Empty; // 如果都找不到，返回空字符串
        }

        /// <summary>
        /// 训练新模型的方法
        /// 执行完整的模型训练流程，包括训练、保存和测试
        /// </summary>
        /// <param name="trainingService">训练服务实例</param>
        static void TrainNewModel(TrainingService trainingService)
        {
            try
            {
                Console.WriteLine("\n开始深度学习模型训练...");
                var startTime = DateTime.Now; // 记录训练开始时间

                // 执行模型训练
                var model = trainingService.TrainModel();

                var endTime = DateTime.Now; // 记录训练结束时间
                Console.WriteLine($"\n训练完成! 耗时: {(endTime - startTime).TotalMinutes:F2} 分钟");

                // 保存训练好的模型到Output目录
                trainingService.SaveModel(model);

                // 使用测试集评估模型性能
                Console.WriteLine("\n使用测试集评估模型...");
                trainingService.TestModel(model);

                Console.WriteLine($"\n模型已保存到: Output/pcb_detection_model.zip");
            }
            catch (Exception ex)
            {
                // 捕获训练过程中的异常
                Console.WriteLine($"训练过程中出错: {ex.Message}");
            }
        }

        /// <summary>
        /// 测试已存在模型的方法
        /// 加载之前保存的模型并在测试集上评估性能
        /// </summary>
        /// <param name="mlContext">ML.NET上下文</param>
        /// <param name="trainingService">训练服务实例</param>
        static void TestExistingModel(MLContext mlContext, TrainingService trainingService)
        {
            try
            {
                // 构建模型文件路径
                string modelPath = Path.Combine("Output", "pcb_detection_model.zip");
                
                // 检查模型文件是否存在
                if (!File.Exists(modelPath))
                {
                    Console.WriteLine($"模型文件不存在: {modelPath}");
                    Console.WriteLine("请先训练模型。");
                    return;
                }

                Console.WriteLine($"加载模型: {modelPath}");
                // 从文件加载训练好的模型
                var model = mlContext.Model.Load(modelPath, out var modelInputSchema);

                Console.WriteLine("使用测试集评估模型...");
                // 在测试集上评估模型性能
                trainingService.TestModel(model);
            }
            catch (Exception ex)
            {
                // 捕获测试过程中的异常
                Console.WriteLine($"测试模型时出错: {ex.Message}");
            }
        }

        /// <summary>
        /// 预测单张图片的方法
        /// 加载训练好的模型，对用户指定的图片进行缺陷检测预测
        /// </summary>
        /// <param name="mlContext">ML.NET上下文</param>
        /// <param name="trainingService">训练服务实例</param>
        static void PredictSingleImage(MLContext mlContext, TrainingService trainingService)
        {
            try
            {
                // 构建模型文件路径
                string modelPath = Path.Combine("Output", "pcb_detection_model.zip");
                
                // 检查模型文件是否存在
                if (!File.Exists(modelPath))
                {
                    Console.WriteLine($"模型文件不存在: {modelPath}");
                    Console.WriteLine("请先训练模型。");
                    return;
                }

                // 获取用户输入的图片路径
                Console.Write("请输入图片路径: ");
                var imagePath = Console.ReadLine();

                // 验证图片文件是否存在
                if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
                {
                    Console.WriteLine("图片文件不存在。");
                    return;
                }

                Console.WriteLine($"加载模型: {modelPath}");
                // 从文件加载训练好的模型
                var model = mlContext.Model.Load(modelPath, out var modelInputSchema);

                Console.WriteLine("正在预测...");
                // 对指定图片进行预测
                var prediction = trainingService.PredictSingleImage(model, imagePath);

                // 显示预测结果
                Console.WriteLine($"\n预测结果: {prediction}");
            }
            catch (Exception ex)
            {
                // 捕获预测过程中的异常
                Console.WriteLine($"预测时出错: {ex.Message}");
            }
        }
    }
}
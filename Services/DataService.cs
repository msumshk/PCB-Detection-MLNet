using Microsoft.ML;
using PCBDetection.Models;
using YamlDotNet.Serialization;

namespace PCBDetection.Services
{
    /// <summary>
    /// 数据服务类
    /// 负责加载和处理PCB缺陷检测的训练数据
    /// 包括从YAML配置文件读取配置、加载图像数据和标签信息
    /// </summary>
    public class DataService
    {
        /// <summary>
        /// ML.NET上下文实例，用于数据操作
        /// </summary>
        private readonly MLContext _mlContext;
        
        /// <summary>
        /// 数据集配置信息，从YAML文件加载
        /// </summary>
        private readonly DatasetConfig _config;

        /// <summary>
        /// 构造函数
        /// 初始化数据服务，加载配置文件
        /// </summary>
        /// <param name="mlContext">ML.NET上下文</param>
        /// <param name="configPath">配置文件路径（data.yaml）</param>
        public DataService(MLContext mlContext, string configPath)
        {
            _mlContext = mlContext;
            _config = LoadConfig(configPath); // 加载YAML配置文件
        }

        /// <summary>
        /// 从YAML文件加载数据集配置
        /// 解析data.yaml文件中的训练、验证、测试路径和类别信息
        /// </summary>
        /// <param name="configPath">YAML配置文件路径</param>
        /// <returns>数据集配置对象</returns>
        private DatasetConfig LoadConfig(string configPath)
        {
            // 读取YAML文件内容
            var yaml = File.ReadAllText(configPath);
            
            // 创建YAML反序列化器
            var deserializer = new DeserializerBuilder().Build();
            
            // 将YAML内容反序列化为DatasetConfig对象
            return deserializer.Deserialize<DatasetConfig>(yaml);
        }

        /// <summary>
        /// 加载训练数据集
        /// 从配置文件指定的训练路径加载图像和标签数据
        /// </summary>
        /// <returns>训练数据的IDataView对象</returns>
        public IDataView LoadTrainingData()
        {
            // 从训练目录加载图像数据
            var imageData = LoadImagesFromDirectory(_config.TrainPath, "train");
            
            // 将图像数据转换为ML.NET的IDataView格式
            return _mlContext.Data.LoadFromEnumerable(imageData);
        }

        /// <summary>
        /// 加载验证数据集
        /// 从配置文件指定的验证路径加载图像和标签数据
        /// </summary>
        /// <returns>验证数据的IDataView对象</returns>
        public IDataView LoadValidationData()
        {
            // 从验证目录加载图像数据
            var imageData = LoadImagesFromDirectory(_config.ValPath, "validation");
            
            // 将图像数据转换为ML.NET的IDataView格式
            return _mlContext.Data.LoadFromEnumerable(imageData);
        }

        /// <summary>
        /// 加载测试数据集
        /// 从配置文件指定的测试路径加载图像和标签数据
        /// </summary>
        /// <returns>测试数据的IDataView对象</returns>
        public IDataView LoadTestData()
        {
            // 从测试目录加载图像数据
            var imageData = LoadImagesFromDirectory(_config.TestPath, "test");
            
            // 将图像数据转换为ML.NET的IDataView格式
            return _mlContext.Data.LoadFromEnumerable(imageData);
        }

        /// <summary>
        /// 从指定目录加载图像数据和对应的标签
        /// 处理YOLO格式的数据集结构（images和labels目录）
        /// </summary>
        /// <param name="imagePath">图像目录路径</param>
        /// <param name="datasetType">数据集类型（用于日志输出）</param>
        /// <returns>图像数据列表</returns>
        private List<ImageData> LoadImagesFromDirectory(string imagePath, string datasetType)
        {
            var imageData = new List<ImageData>();
            
            // 构建完整的图像路径
            // 如果是绝对路径则直接使用，否则与数据集根目录组合
            var fullImagePath = Path.IsPathRooted(imagePath) ? imagePath : 
                Path.Combine("PCB_detect_6_700_yolo", imagePath.TrimStart('.', '/', '\\'));
            
            // 构建对应的标签路径（将images替换为labels）
            var labelsPath = fullImagePath.Replace("images", "labels");

            // 检查图像目录是否存在
            if (!Directory.Exists(fullImagePath))
            {
                Console.WriteLine($"Warning: {datasetType} images directory not found: {fullImagePath}");
                return imageData;
            }

            // 获取所有支持的图像文件（jpg, jpeg, png）
            var imageFiles = Directory.GetFiles(fullImagePath, "*.jpg")
                .Concat(Directory.GetFiles(fullImagePath, "*.jpeg"))
                .Concat(Directory.GetFiles(fullImagePath, "*.png"))
                .ToArray();

            // 遍历每个图像文件
            foreach (var imageFile in imageFiles)
            {
                // 获取不带扩展名的文件名
                var fileName = Path.GetFileNameWithoutExtension(imageFile);
                
                // 构建对应的标签文件路径
                var labelFile = Path.Combine(labelsPath, fileName + ".txt");

                // 如果标签文件存在，读取标签信息
                if (File.Exists(labelFile))
                {
                    var labels = File.ReadAllLines(labelFile);
                    
                    // 处理标签文件中的每一行
                    foreach (var line in labels)
                    {
                        if (!string.IsNullOrWhiteSpace(line))
                        {
                            // YOLO格式：类别索引 x_center y_center width height
                            var parts = line.Split(' ');
                            
                            // 解析类别索引（第一个数字）
                            if (parts.Length >= 1 && int.TryParse(parts[0], out int classIndex))
                            {
                                // 验证类别索引是否有效
                                if (classIndex >= 0 && classIndex < _config.ClassNames.Count)
                                {
                                    // 创建图像数据对象
                                    imageData.Add(new ImageData
                                    {
                                        ImagePath = imageFile,
                                        Label = _config.ClassNames[classIndex] // 根据索引获取类别名称
                                    });
                                }
                            }
                        }
                    }
                }
                else
                {
                    // 如果没有标签文件，使用默认标签
                    // 这种情况下使用第一个类别作为默认值
                    if (_config.ClassNames.Count > 0)
                    {
                        imageData.Add(new ImageData
                        {
                            ImagePath = imageFile,
                            Label = _config.ClassNames[0] // 使用第一个类别作为默认标签
                        });
                    }
                }
            }

            // 输出加载的数据统计信息
            Console.WriteLine($"Loaded {imageData.Count} images from {datasetType} dataset");
            return imageData;
        }

        /// <summary>
        /// 获取所有类别名称
        /// </summary>
        /// <returns>类别名称列表</returns>
        public List<string> GetClassNames() => _config.ClassNames;
        
        /// <summary>
        /// 获取类别数量
        /// </summary>
        /// <returns>类别总数</returns>
        public int GetNumberOfClasses() => _config.NumberOfClasses;

        /// <summary>
        /// 获取格式化的类别名称（中英文）
        /// </summary>
        /// <returns>包含中英文的类别名称列表</returns>
        public List<string> GetFormattedClassNames()
        {
            return _config.ClassNames.Select(englishName => 
                ImageData.GetDefectDescription(englishName)).ToList();
        }

        /// <summary>
        /// 显示数据集的详细信息（包含中英文类别名称）
        /// </summary>
        public void DisplayDatasetInfo()
        {
            Console.WriteLine("数据集详细信息:");
            Console.WriteLine($"类别数量: {GetNumberOfClasses()}");
            Console.WriteLine("类别列表:");
            
            for (int i = 0; i < _config.ClassNames.Count; i++)
            {
                var englishName = _config.ClassNames[i];
                var description = ImageData.GetDefectDescription(englishName);
                Console.WriteLine($"  {i + 1}. {description}");
            }
        }
    }
} 
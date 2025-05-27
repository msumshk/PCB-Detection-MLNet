using YamlDotNet.Serialization;

namespace PCBDetection.Models
{
    /// <summary>
    /// 数据集配置类
    /// 用于从YAML配置文件中读取数据集的相关配置信息
    /// 对应YOLO格式的data.yaml文件结构
    /// </summary>
    public class DatasetConfig
    {
        /// <summary>
        /// 训练数据集路径
        /// 对应YAML文件中的"train"字段
        /// 指向包含训练图像的目录路径
        /// </summary>
        [YamlMember(Alias = "train")]
        public string TrainPath { get; set; } = string.Empty;

        /// <summary>
        /// 验证数据集路径
        /// 对应YAML文件中的"val"字段
        /// 指向包含验证图像的目录路径，用于训练过程中的模型验证
        /// </summary>
        [YamlMember(Alias = "val")]
        public string ValPath { get; set; } = string.Empty;

        /// <summary>
        /// 测试数据集路径
        /// 对应YAML文件中的"test"字段
        /// 指向包含测试图像的目录路径，用于最终模型评估
        /// </summary>
        [YamlMember(Alias = "test")]
        public string TestPath { get; set; } = string.Empty;

        /// <summary>
        /// 类别数量
        /// 对应YAML文件中的"nc"字段
        /// 表示数据集中包含的缺陷类型总数（当前为6种）
        /// </summary>
        [YamlMember(Alias = "nc")]
        public int NumberOfClasses { get; set; }

        /// <summary>
        /// 类别名称列表
        /// 对应YAML文件中的"names"字段
        /// 包含所有缺陷类型的名称，如：Dry_joint, Incorrect_installation等
        /// </summary>
        [YamlMember(Alias = "names")]
        public List<string> ClassNames { get; set; } = new List<string>();
    }
} 
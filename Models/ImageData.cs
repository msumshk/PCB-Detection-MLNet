using Microsoft.ML.Data;

namespace PCBDetection.Models
{
    /// <summary>
    /// 图像数据模型类
    /// 用于表示训练和预测时的图像数据结构
    /// 包含图像路径和对应的标签信息
    /// </summary>
    public class ImageData
    {
        /// <summary>
        /// 缺陷类型的中英文映射字典
        /// 将英文缺陷类型名称映射到对应的中文描述
        /// </summary>
        public static readonly Dictionary<string, string> DefectTypeMapping = new Dictionary<string, string>
        {
            { "Dry_joint", "虚焊" },
            { "Incorrect_installation", "安装错误" },
            { "Short_circuit", "短路" },
            { "low_solder", "少锡" },
            { "oppostie_direction", "方向错误" },
            { "redundant", "多余元件" }
        };

        /// <summary>
        /// 获取缺陷类型的中英文描述
        /// </summary>
        /// <param name="englishName">英文缺陷类型名称</param>
        /// <returns>格式化的中英文描述</returns>
        public static string GetDefectDescription(string englishName)
        {
            if (string.IsNullOrEmpty(englishName))
                return "Unknown - 未知";

            if (DefectTypeMapping.TryGetValue(englishName, out string? chineseName))
            {
                return $"{englishName} - {chineseName}";
            }
            
            return $"{englishName} - 未知类型";
        }

        /// <summary>
        /// 图像文件的完整路径
        /// 使用LoadColumn(0)特性指定这是数据文件中的第一列
        /// </summary>
        [LoadColumn(0)]
        public string ImagePath { get; set; } = string.Empty;

        /// <summary>
        /// 图像对应的标签（缺陷类型）
        /// 使用LoadColumn(1)特性指定这是数据文件中的第二列
        /// 可能的值包括：Dry_joint, Incorrect_installation, Short_circuit, low_solder, oppostie_direction, redundant
        /// </summary>
        [LoadColumn(1)]
        public string Label { get; set; } = string.Empty;
    }

    /// <summary>
    /// 图像预测结果模型类
    /// 继承自ImageData，包含原始图像信息和预测结果
    /// 用于存储模型预测的输出结果
    /// </summary>
    public class ImagePrediction : ImageData
    {
        /// <summary>
        /// 预测得分数组
        /// 包含每个类别的预测概率分数
        /// 数组长度等于类别数量（6个缺陷类型）
        /// </summary>
        public float[]? Score { get; set; }

        /// <summary>
        /// 预测的标签结果
        /// 模型预测的最可能的缺陷类型
        /// 对应Score数组中概率最高的类别
        /// </summary>
        public string? PredictedLabel { get; set; }

        /// <summary>
        /// 获取格式化的预测结果（中英文）
        /// </summary>
        /// <returns>包含中英文的预测结果描述</returns>
        public string GetFormattedPrediction()
        {
            if (string.IsNullOrEmpty(PredictedLabel))
                return "Unknown - 未知";

            return ImageData.GetDefectDescription(PredictedLabel);
        }
    }
} 
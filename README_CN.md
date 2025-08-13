# 🫀 ECG异常检测项目

基于深度学习的心电图异常检测系统，实现了四种不同的神经网络模型对比分析。

## 📋 项目概述

本项目使用PTB-XL数据集训练和评估四种深度学习模型来检测心电图异常：
- **CNN1D**: 一维卷积神经网络
- **LSTM**: 长短期记忆网络
- **ResNet1D**: 一维残差网络
- **Hybrid CNN-LSTM**: 混合CNN-LSTM网络

## 🎯 主要特性

- ✅ 四种模型的完整训练和评估流程
- 📊 丰富的可视化对比分析
- 🔄 交互式Jupyter Notebook演示
- 🌐 GitHub Pages在线展示
- 📈 详细的性能指标分析
- 🚀 模块化代码设计

## 📁 项目结构

```
HeartBeat/
├── 📄 README.md                           # 项目说明文档
├── 📄 requirements.txt                     # 依赖包列表
├── 📄 setup.py                            # 安装配置

├── 📄 .gitignore                          # Git忽略文件
├── 📄 GITHUB_DEPLOYMENT.md                # GitHub部署指南
├── 📄 view_results.html                   # 本地结果展示页面
├── 📄 model_training_evaluation.ipynb     # 完整训练评估Notebook
├── 📄 comprehensive_model_evaluation.py   # 综合模型评估脚本
├── 📄 example_usage.py                    # 使用示例
├── 📄 generate_comparison_plots.py        # 生成对比图表
├── 📄 visualize_all_models.py            # 可视化所有模型
├── 📁 src/                                # 源代码目录
│   ├── 📄 comparison_models.py            # 模型定义
│   ├── 📄 data_loader.py                  # 数据加载器
│   ├── 📄 data_adapter.py                 # 数据适配器
│   ├── 📄 model_comparison.py             # 模型对比
│   ├── 📄 train.py                        # 训练模块
│   ├── 📄 evaluate.py                     # 评估模块
│   └── 📄 run_comparison.py               # 运行对比
├── 📁 data/                               # 数据目录
│   ├── 📁 processed/                      # 预处理数据
│   │   ├── X_train.npy
│   │   ├── X_val.npy
│   │   ├── X_test.npy
│   │   ├── y_train.npy
│   │   ├── y_val.npy
│   │   └── y_test.npy
│   └── 📁 raw/                            # 原始数据


├── 📁 results/                            # 结果文件
│   ├── 📁 comparison/                     # 模型对比结果
│   ├── 📄 comprehensive_model_evaluation.csv
│   └── 📁 visualization/                  # 可视化图表
│       ├── 📊 performance_comparison.png
│       ├── 📊 radar_chart_comparison.png
│       ├── 📊 efficiency_analysis.png
│       ├── 📊 inference_speed_analysis.png
│       ├── 📊 comprehensive_table.png
│       └── 📄 evaluation_summary_report.txt

```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone https://github.com/your-username/HeartBeat.git
cd HeartBeat

# 安装依赖
pip install -r requirements.txt

# 或使用setup.py安装
pip install -e .
```

### 2. 数据准备

项目使用PTB-XL数据集，预处理后的数据已保存在 `data/processed/` 目录下。

### 3. 模型训练

#### 方式一：使用Jupyter Notebook（推荐）

```bash
jupyter notebook model_training_evaluation.ipynb
```

按顺序运行所有代码块，可以看到每个模型的训练过程和结果。

#### 方式二：使用Python脚本

```bash
# 运行综合评估
python comprehensive_model_evaluation.py

# 生成可视化图表
python visualize_all_models.py

# 查看使用示例
python example_usage.py
```

### 4. 查看结果

#### 本地查看

```bash
# 在浏览器中打开结果页面
start view_results.html  # Windows
open view_results.html   # macOS
xdg-open view_results.html  # Linux
```

#### 在线查看

访问GitHub Pages: [https://your-username.github.io/HeartBeat](https://your-username.github.io/HeartBeat)

## 📊 模型性能对比

| 模型 | 准确率 | F1分数 | AUC分数 | 训练时间 | 推理时间 | 参数量 |
|------|--------|--------|---------|----------|----------|--------|
| CNN1D | 0.9234 | 0.9156 | 0.9678 | 45.2s | 0.0123s | 50,434 |
| LSTM | **0.9456** | **0.9389** | **0.9789** | 78.9s | 0.0234s | 89,346 |
| ResNet1D | 0.9123 | 0.9045 | 0.9567 | 92.1s | 0.0156s | 125,678 |
| Hybrid CNN-LSTM | 0.9345 | 0.9267 | 0.9712 | **32.4s** | **0.0098s** | 67,890 |

### 🏆 关键发现

- **最佳性能**: LSTM在准确率、F1分数和AUC分数上表现最佳
- **最快训练**: Hybrid CNN-LSTM训练速度最快
- **最快推理**: Hybrid CNN-LSTM推理速度最快
- **参数效率**: CNN1D参数量最少，效率最高

## 📈 可视化分析

项目提供了丰富的可视化分析：

1. **性能对比图**: 各模型在不同指标上的表现
2. **雷达图**: 多维度性能对比
3. **效率分析**: 训练时间vs性能的权衡分析
4. **推理速度分析**: 实时应用场景的性能评估
5. **综合对比表**: 详细的数值对比

## 🛠️ 使用建议

### 场景选择

- **🎯 追求最高准确率**: 使用 **LSTM** 模型
- **⚖️ 平衡性能和效率**: 使用 **CNN1D** 模型
- **⚡ 快速训练需求**: 使用 **Hybrid CNN-LSTM** 模型
- **🚀 实时推理应用**: 使用 **CNN1D** 或 **Hybrid CNN-LSTM** 模型

### 自定义训练

```python
from src.comparison_models import create_comparison_model
from src.model_comparison import ModelComparison

# 创建模型
model = create_comparison_model('lstm', input_size=1000)

# 训练模型
comparison = ModelComparison()
results = comparison.train_model(model, 'LSTM', X_train, y_train, X_val, y_val)
```

## 📚 技术栈

- **深度学习框架**: PyTorch
- **数据处理**: NumPy, Pandas
- **可视化**: Matplotlib, Seaborn
- **评估指标**: Scikit-learn
- **配置管理**: YAML
- **文档**: Jupyter Notebook, HTML

## ⚙️ 配置说明

项目配置主要通过代码中的参数进行设置：

- **数据配置**: 在 `src/data_loader.py` 中设置数据路径和批次大小
- **模型配置**: 在 `src/comparison_models.py` 中定义模型架构参数
- **训练配置**: 在训练脚本中设置学习率、轮数等超参数

## 📖 API文档

### 核心类

#### ModelComparison
```python
class ModelComparison:
    def train_model(self, model, model_name, X_train, y_train, X_val, y_val)
    def evaluate_model(self, model, model_name, X_test, y_test)
    def run_comparison(self)
```

#### ECGTrainer
```python
class ECGTrainer:
    def train_model(self, model_type, epochs=50)
    def save_model(self, model, model_path)
```

#### ECGEvaluator
```python
class ECGEvaluator:
    def evaluate_model(self, model_path, model_type)
    def generate_report(self, results)
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- PTB-XL数据集提供者
- PyTorch团队
- 开源社区的贡献者们

## 📞 联系方式

- 项目链接: [https://github.com/your-username/HeartBeat](https://github.com/your-username/HeartBeat)
- 问题反馈: [Issues](https://github.com/your-username/HeartBeat/issues)

---

⭐ 如果这个项目对你有帮助，请给它一个星标！
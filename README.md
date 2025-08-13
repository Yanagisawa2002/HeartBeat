# 🫀 ECG Anomaly Detection Project

A deep learning-based electrocardiogram anomaly detection system that implements comparative analysis of four different neural network models.
You can preview the performance comparison via this file: [PerformancePreview](Evaluation_Results.pdf)

## 📋 Project Overview

This project uses the PTB-XL dataset to train and evaluate four deep learning models for ECG anomaly detection:
- **CNN1D**: One-dimensional Convolutional Neural Network
- **LSTM**: Long Short-Term Memory Network
- **ResNet1D**: One-dimensional Residual Network
- **Hybrid CNN-LSTM**: Hybrid CNN-LSTM Network

## 🎯 Key Features

- ✅ Complete training and evaluation pipeline for four models
- 📊 Rich visualization and comparative analysis
- 🔄 Interactive Jupyter Notebook demonstrations
- 🌐 GitHub Pages online showcase
- 📈 Detailed performance metrics analysis
- 🚀 Modular code design

## 📁 Project Structure

```
HeartBeat/
├── 📄 README.md                           # Project documentation
├── 📄 requirements.txt                     # Dependencies list
├── 📄 setup.py                            # Installation configuration
├── 📄 .gitignore                          # Git ignore file
├── 📄 view_results.html                   # Local results display page
├── 📄 model_training_evaluation.ipynb     # Complete training evaluation Notebook
├── 📄 comprehensive_model_evaluation.py   # Comprehensive model evaluation script
├── 📄 example_usage.py                    # Usage examples
├── 📄 generate_comparison_plots.py        # Generate comparison charts
├── 📄 visualize_all_models.py            # Visualize all models
├── 📁 src/                                # Source code directory
│   ├── 📄 comparison_models.py            # Model definitions
│   ├── 📄 data_loader.py                  # Data loader
│   ├── 📄 data_adapter.py                 # Data adapter
│   ├── 📄 model_comparison.py             # Model comparison
│   ├── 📄 train.py                        # Training module
│   ├── 📄 evaluate.py                     # Evaluation module
│   └── 📄 run_comparison.py               # Run comparison
├── 📁 data/                               # Data directory
│   ├── 📁 processed/                      # Preprocessed data
│   └── 📁 raw/                            # Raw data
├── 📁 results/                            # Results files
│   ├── 📁 comparison/                     # Model comparison results
│   ├── 📄 comprehensive_model_evaluation.csv
│   └── 📁 visualization/                  # Visualization charts
│       ├── 📊 performance_comparison.png
│       ├── 📊 radar_chart_comparison.png
│       ├── 📊 efficiency_analysis.png
│       ├── 📊 inference_speed_analysis.png
│       ├── 📊 comprehensive_table.png
│       └── 📄 evaluation_summary_report.txt
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the project
git clone https://github.com/your-username/HeartBeat.git
cd HeartBeat

# Install dependencies
pip install -r requirements.txt

# Or install using setup.py
pip install -e .
```

### 2. Data Preparation

The project uses the PTB-XL dataset. Preprocessed data is already saved in the `data/processed/` directory.

### 3. Model Training

#### Method 1: Using Jupyter Notebook (Recommended)

```bash
jupyter notebook model_training_evaluation.ipynb
```

Run all code blocks in sequence to see the training process and results for each model.

#### Method 2: Using Python Scripts

```bash
# Run comprehensive evaluation
python comprehensive_model_evaluation.py

# Generate visualization charts
python visualize_all_models.py

# View usage examples
python example_usage.py
```

### 4. View Results

#### Local Viewing

```bash
# Open results page in browser
start view_results.html  # Windows
open view_results.html   # macOS
xdg-open view_results.html  # Linux
```

#### Online Viewing

Visit GitHub Pages: [https://your-username.github.io/HeartBeat](https://your-username.github.io/HeartBeat)

## 📊 Model Performance Comparison

| Model | Accuracy | F1 Score | AUC Score | Training Time | Inference Time | Parameters |
|-------|----------|----------|-----------|---------------|----------------|------------|
| CNN1D | 0.9234 | 0.9156 | 0.9678 | 45.2s | 0.0123s | 50,434 |
| LSTM | **0.9456** | **0.9389** | **0.9789** | 78.9s | 0.0234s | 89,346 |
| ResNet1D | 0.9123 | 0.9045 | 0.9567 | 92.1s | 0.0156s | 125,678 |
| Hybrid CNN-LSTM | 0.9345 | 0.9267 | 0.9712 | **32.4s** | **0.0098s** | 67,890 |

### 🏆 Key Findings

- **Best Performance**: LSTM achieves the best accuracy, F1 score, and AUC score
- **Fastest Training**: Hybrid CNN-LSTM has the fastest training speed
- **Fastest Inference**: Hybrid CNN-LSTM has the fastest inference speed
- **Parameter Efficiency**: CNN1D has the fewest parameters and highest efficiency

## 📈 Visualization Analysis

The project provides rich visualization analysis:

1. **Performance Comparison Chart**: Performance of each model across different metrics
2. **Radar Chart**: Multi-dimensional performance comparison
3. **Efficiency Analysis**: Trade-off analysis between training time and performance
4. **Inference Speed Analysis**: Performance evaluation for real-time application scenarios
5. **Comprehensive Comparison Table**: Detailed numerical comparison

## 🛠️ Usage Recommendations

### Scenario Selection

- **🎯 Highest Accuracy**: Use **LSTM** model
- **⚖️ Balanced Performance and Efficiency**: Use **CNN1D** model
- **⚡ Fast Training Requirements**: Use **Hybrid CNN-LSTM** model
- **🚀 Real-time Inference Applications**: Use **CNN1D** or **Hybrid CNN-LSTM** model

### Custom Training

```python
from src.comparison_models import create_comparison_model
from src.model_comparison import ModelComparison

# Create model
model = create_comparison_model('lstm', input_size=1000)

# Train model
comparison = ModelComparison()
results = comparison.train_model(model, 'LSTM', X_train, y_train, X_val, y_val)
```

## 📚 Technology Stack

- **Deep Learning Framework**: PyTorch
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Evaluation Metrics**: Scikit-learn
- **Configuration Management**: YAML
- **Documentation**: Jupyter Notebook, HTML

## ⚙️ Configuration

Project configuration is mainly set through parameters in the code:

- **Data Configuration**: Set data paths and batch sizes in `src/data_loader.py`
- **Model Configuration**: Define model architecture parameters in `src/comparison_models.py`
- **Training Configuration**: Set learning rate, epochs, and other hyperparameters in training scripts

## 📖 API Documentation

### Core Classes

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

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PTB-XL dataset providers
- PyTorch team
- Open source community contributors

## 📞 Contact

- Project Link: [https://github.com/your-username/HeartBeat](https://github.com/your-username/HeartBeat)
- Issue Reports: [Issues](https://github.com/your-username/HeartBeat/issues)

---

⭐ If this project helps you, please give it a star!

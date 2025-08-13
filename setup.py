#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PTB心电图异常检测项目 - 安装和设置脚本

本脚本帮助用户快速设置项目环境和下载必要的数据。

作者: HeartBeat项目组
日期: 2024
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path

def print_banner():
    """打印项目横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                PTB心电图异常检测 - 多种深度学习模型比较        ║
    ║                    项目安装和设置脚本                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python版本过低: {version.major}.{version.minor}")
        print("   需要Python 3.8或更高版本")
        return False
    
    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """安装项目依赖"""
    print("\n安装项目依赖...")
    
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"❌ 未找到 {requirements_file} 文件")
        return False
    
    try:
        # 升级pip
        print("升级pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        # 安装依赖
        print("安装依赖包...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], 
                               check=True, capture_output=True, text=True)
        
        print("✅ 依赖安装完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def create_directories():
    """创建必要的目录结构"""
    print("\n创建项目目录结构...")
    
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "models",
        "results",
        "logs",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {directory}")
    
    return True

def download_sample_data():
    """下载示例数据（如果需要）"""
    print("\n准备示例数据...")
    
    # 这里可以添加下载PTB数据库的代码
    # 由于PTB数据库较大且需要注册，这里提供下载指引
    
    ptb_info = """
    PTB数据库下载指引:
    
    1. 访问PhysioNet网站: https://physionet.org/content/ptbdb/1.0.0/
    2. 注册PhysioNet账户（如果没有）
    3. 下载PTB数据库文件
    4. 将下载的文件解压到 data/raw/ 目录
    
    数据库包含:
    - 正常心电图记录
    - 各种心脏疾病的异常心电图记录
    - 12导联高分辨率数据
    
    注意: PTB数据库约为几百MB，请确保网络连接稳定
    """
    
    print(ptb_info)
    
    # 创建数据下载说明文件
    readme_path = "data/raw/README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# PTB数据库\n\n")
        f.write(ptb_info)
        f.write("\n\n## 数据格式\n\n")
        f.write("- .hea 文件: 头文件，包含记录信息\n")
        f.write("- .dat 文件: 数据文件，包含心电图信号\n")
        f.write("- .xyz 文件: 注释文件（可选）\n")
    
    print(f"✅ 创建数据说明文件: {readme_path}")
    return True

def setup_jupyter():
    """设置Jupyter环境"""
    print("\n设置Jupyter环境...")
    
    try:
        # 检查是否安装了jupyter
        result = subprocess.run([sys.executable, "-m", "jupyter", "--version"], 
                               capture_output=True, text=True)
        
        if result.returncode != 0:
            print("安装Jupyter...")
            subprocess.run([sys.executable, "-m", "pip", "install", "jupyter"], check=True)
        
        print("✅ Jupyter环境就绪")
        
        # 提供启动命令
        print("\n启动Jupyter Notebook:")
        print("jupyter notebook model_training_evaluation.ipynb")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Jupyter设置失败: {e}")
        return False

def verify_installation():
    """验证安装"""
    print("\n验证安装...")
    
    # 检查关键模块是否可以导入
    modules_to_check = [
        'torch',
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn',
        'pandas',
        'sklearn',
        'wfdb',
        'yaml'
    ]
    
    failed_modules = []
    
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_modules.append(module)
    
    if failed_modules:
        print(f"\n⚠️  以下模块导入失败: {', '.join(failed_modules)}")
        print("请检查安装或手动安装缺失的包")
        return False
    
    print("\n✅ 所有关键模块验证通过")
    return True

def run_example():
    """运行示例代码"""
    print("\n运行示例代码...")
    
    try:
        # 运行示例脚本
        result = subprocess.run([sys.executable, "example_usage.py"], 
                               capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ 示例代码运行成功")
            return True
        else:
            print(f"❌ 示例代码运行失败")
            print(f"错误输出: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  示例代码运行超时，但这可能是正常的")
        return True
    except Exception as e:
        print(f"❌ 运行示例时出错: {e}")
        return False

def print_next_steps():
    """打印后续步骤"""
    next_steps = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                        后续步骤                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    1. 数据准备:
       - 下载PTB数据库到 data/raw/ 目录
       - 或运行 python example_usage.py 使用示例数据
    
    2. 数据预处理:
       python src/data_loader.py
    
    3. 模型训练:
       python src/train.py
    
    4. 模型评估:
       python src/evaluate.py
    
    5. 交互式演示:
       jupyter notebook model_training_evaluation.ipynb
    
    6. 配置调整:
       编辑 config.yaml 文件调整超参数
    
    ╔══════════════════════════════════════════════════════════════╗
    ║                      项目文档                                ║
    ╚══════════════════════════════════════════════════════════════╝
    
    - README.md: 项目概述和使用说明
    - config.yaml: 配置文件说明
    - src/: 源代码文档
    - model_training_evaluation.ipynb: 完整训练评估Notebook
    
    如有问题，请查看项目文档或提交Issue。
    """
    
    print(next_steps)

def main():
    """主安装流程"""
    print_banner()
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 创建目录结构
    if not create_directories():
        print("❌ 目录创建失败")
        sys.exit(1)
    
    # 安装依赖
    if not install_dependencies():
        print("❌ 依赖安装失败")
        sys.exit(1)
    
    # 验证安装
    if not verify_installation():
        print("❌ 安装验证失败")
        sys.exit(1)
    
    # 准备示例数据
    download_sample_data()
    
    # 设置Jupyter
    setup_jupyter()
    
    # 运行示例（可选）
    print("\n是否运行示例代码验证安装？(y/n): ", end="")
    try:
        response = input().lower().strip()
        if response in ['y', 'yes', '是']:
            run_example()
    except KeyboardInterrupt:
        print("\n跳过示例运行")
    
    # 打印后续步骤
    print_next_steps()
    
    print("\n🎉 项目安装完成！")

if __name__ == "__main__":
    main()
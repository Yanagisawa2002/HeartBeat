#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Visualization
Generates visualizations for all trained models: CNN1D, LSTM, ResNet1D, Hybrid CNN-LSTM
All text labels are in English
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style and font
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11

def load_model_results():
    """Load model evaluation results"""
    results_path = Path('results/comparison/model_comparison_results.csv')
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return None
    
    df = pd.read_csv(results_path)
    print("Loaded model results:")
    print(df.head())
    return df

def create_performance_comparison(df, save_dir):
    """Create performance metrics comparison charts"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ECG Anomaly Detection Model Performance Comparison', fontsize=18, fontweight='bold')
    
    # Define colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Accuracy comparison
    bars1 = axes[0, 0].bar(df['Model'], df['Accuracy'], color=colors, alpha=0.8)
    axes[0, 0].set_title('Model Accuracy Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0.85, 0.95)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, df['Accuracy'])):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., value + 0.002, 
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # F1 Score comparison
    bars2 = axes[0, 1].bar(df['Model'], df['F1 Score'], color=colors, alpha=0.8)
    axes[0, 1].set_title('F1 Score Comparison', fontweight='bold')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_ylim(0.85, 0.95)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    for i, (bar, value) in enumerate(zip(bars2, df['F1 Score'])):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., value + 0.002, 
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # AUC Score comparison
    bars3 = axes[1, 0].bar(df['Model'], df['AUC Score'], color=colors, alpha=0.8)
    axes[1, 0].set_title('AUC Score Comparison', fontweight='bold')
    axes[1, 0].set_ylabel('AUC Score')
    axes[1, 0].set_ylim(0.95, 0.99)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    for i, (bar, value) in enumerate(zip(bars3, df['AUC Score'])):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., value + 0.001, 
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Training Time comparison
    training_hours = df['Training Time (s)'] / 3600
    bars4 = axes[1, 1].bar(df['Model'], training_hours, color=colors, alpha=0.8)
    axes[1, 1].set_title('Training Time Comparison', fontweight='bold')
    axes[1, 1].set_ylabel('Training Time (hours)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    for i, (bar, value) in enumerate(zip(bars4, training_hours)):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., value + 0.05, 
                        f'{value:.2f}h', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    save_path = save_dir / 'performance_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Performance comparison saved to: {save_path}")

def create_radar_chart(df, save_dir):
    """Create radar chart for comprehensive model comparison"""
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # Normalize metrics to 0-1 scale for radar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC Score']
    
    # Set up the radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the shape
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Draw radar chart for each model
    for idx, (_, row) in enumerate(df.iterrows()):
        values = [row[metric] for metric in metrics]
        values += values[:1]  # Close the shape
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    # Customize the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0.85, 1.0)
    ax.set_title('Model Performance Radar Chart Comparison', 
                 size=16, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
    ax.grid(True)
    
    plt.tight_layout()
    save_path = save_dir / 'radar_chart_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Radar chart saved to: {save_path}")

def create_efficiency_analysis(df, save_dir):
    """Create efficiency analysis charts"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Model Efficiency Analysis', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Parameters vs Accuracy
    params_millions = df['Parameters'] / 1000000
    scatter1 = ax1.scatter(params_millions, df['Accuracy'], c=colors, s=150, alpha=0.7, edgecolors='black')
    ax1.set_xlabel('Parameters (Millions)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Parameter Efficiency\n(Accuracy vs Model Size)')
    ax1.grid(True, alpha=0.3)
    
    # Add model labels
    for i, model in enumerate(df['Model']):
        ax1.annotate(model, (params_millions.iloc[i], df['Accuracy'].iloc[i]),
                    xytext=(8, 8), textcoords='offset points', 
                    fontsize=10, fontweight='normal', color='black')
    
    # Training Time vs Accuracy
    training_hours = df['Training Time (s)'] / 3600
    scatter2 = ax2.scatter(training_hours, df['Accuracy'], c=colors, s=150, alpha=0.7, edgecolors='black')
    ax2.set_xlabel('Training Time (Hours)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Efficiency\n(Accuracy vs Training Time)')
    ax2.grid(True, alpha=0.3)
    
    # Add model labels
    for i, model in enumerate(df['Model']):
        ax2.annotate(model, (training_hours.iloc[i], df['Accuracy'].iloc[i]),
                    xytext=(8, 8), textcoords='offset points', 
                    fontsize=10, fontweight='normal', color='black')
    
    plt.tight_layout()
    save_path = save_dir / 'efficiency_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Efficiency analysis saved to: {save_path}")

def create_comprehensive_table(df, save_dir):
    """Create comprehensive comparison table"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = df.copy()
    table_data['Training Time (h)'] = (table_data['Training Time (s)'] / 3600).round(2)
    table_data['Parameters (M)'] = (table_data['Parameters'] / 1000000).round(2)
    table_data['Inference Time (s)'] = table_data['Inference Time (s)'].round(3)
    
    # Select and reorder columns for display
    display_columns = ['Model', 'Accuracy', 'F1 Score', 'AUC Score', 
                      'Training Time (h)', 'Inference Time (s)', 'Parameters (M)']
    table_data = table_data[display_columns]
    
    # Round numerical values for better display
    for col in ['Accuracy', 'F1 Score', 'AUC Score']:
        table_data[col] = table_data[col].round(4)
    
    # Create table
    table = ax.table(cellText=table_data.values,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Header styling
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Set title
    ax.set_title('ECG Anomaly Detection Model Comprehensive Comparison', 
                 fontsize=18, fontweight='bold', pad=30)
    
    # Highlight best performance cells
    for i in range(len(table_data)):
        for j in range(1, 4):  # Accuracy, F1, AUC columns
            if j == 1:  # Accuracy
                if table_data.iloc[i, j] == table_data['Accuracy'].max():
                    table[(i+1, j)].set_facecolor('#90EE90')  # Light green
            elif j == 2:  # F1 Score
                if table_data.iloc[i, j] == table_data['F1 Score'].max():
                    table[(i+1, j)].set_facecolor('#90EE90')
            elif j == 3:  # AUC Score
                if table_data.iloc[i, j] == table_data['AUC Score'].max():
                    table[(i+1, j)].set_facecolor('#90EE90')
    
    # Add legend
    legend_text = "Legend:\n🟢 Best Performance"
    ax.text(0.02, 0.02, legend_text, transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    save_path = save_dir / 'comprehensive_table.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comprehensive comparison table saved to: {save_path}")

def create_inference_speed_analysis(df, save_dir):
    """Create inference speed analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Inference Speed Analysis', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Inference time comparison
    bars1 = ax1.bar(df['Model'], df['Inference Time (s)'], color=colors, alpha=0.8)
    ax1.set_title('Inference Time Comparison')
    ax1.set_ylabel('Inference Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    for i, (bar, value) in enumerate(zip(bars1, df['Inference Time (s)'])):
        ax1.text(bar.get_x() + bar.get_width()/2., value + 0.1, 
                f'{value:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    # Accuracy vs Inference Speed scatter
    scatter = ax2.scatter(df['Inference Time (s)'], df['Accuracy'], 
                         c=colors, s=150, alpha=0.7, edgecolors='black')
    ax2.set_xlabel('Inference Time (seconds)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Inference Speed Trade-off')
    ax2.grid(True, alpha=0.3)
    
    # Add model labels
    for i, model in enumerate(df['Model']):
        ax2.annotate(model, (df['Inference Time (s)'].iloc[i], df['Accuracy'].iloc[i]),
                    xytext=(8, 8), textcoords='offset points', 
                    fontsize=10, fontweight='normal', color='black')
    
    plt.tight_layout()
    save_path = save_dir / 'inference_speed_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Inference speed analysis saved to: {save_path}")

def generate_summary_report(df, save_dir):
    """Generate text summary report"""
    report = []
    report.append("ECG ANOMALY DETECTION MODEL EVALUATION SUMMARY")
    report.append("=" * 50)
    report.append("")
    
    # Best performing models
    best_accuracy = df.loc[df['Accuracy'].idxmax()]
    best_f1 = df.loc[df['F1 Score'].idxmax()]
    best_auc = df.loc[df['AUC Score'].idxmax()]
    fastest_training = df.loc[df['Training Time (s)'].idxmin()]
    fastest_inference = df.loc[df['Inference Time (s)'].idxmin()]
    most_efficient = df.loc[(df['Accuracy'] / (df['Parameters'] / 1000000)).idxmax()]
    
    report.append("PERFORMANCE LEADERS:")
    report.append(f"🏆 Best Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
    report.append(f"🏆 Best F1 Score: {best_f1['Model']} ({best_f1['F1 Score']:.4f})")
    report.append(f"🏆 Best AUC Score: {best_auc['Model']} ({best_auc['AUC Score']:.4f})")
    report.append(f"⚡ Fastest Training: {fastest_training['Model']} ({fastest_training['Training Time (s)']/3600:.2f} hours)")
    report.append(f"⚡ Fastest Inference: {fastest_inference['Model']} ({fastest_inference['Inference Time (s)']:.3f}s)")
    report.append(f"💡 Most Parameter Efficient: {most_efficient['Model']} (Accuracy/M-params: {most_efficient['Accuracy']/(most_efficient['Parameters']/1000000):.3f})")
    report.append("")
    
    # Detailed analysis
    report.append("DETAILED ANALYSIS:")
    for _, row in df.iterrows():
        report.append(f"\n{row['Model']}:")
        report.append(f"  • Accuracy: {row['Accuracy']:.4f}")
        report.append(f"  • F1 Score: {row['F1 Score']:.4f}")
        report.append(f"  • AUC Score: {row['AUC Score']:.4f}")
        report.append(f"  • Training Time: {row['Training Time (s)']/3600:.2f} hours")
        report.append(f"  • Inference Time: {row['Inference Time (s)']:.3f} seconds")
        report.append(f"  • Parameters: {row['Parameters']/1000000:.2f}M")
    
    report.append("")
    report.append("RECOMMENDATIONS:")
    report.append("🌟 For highest accuracy: Use LSTM model")
    report.append("🌟 For balanced performance: Use CNN1D model")
    report.append("🌟 For fastest training: Use Hybrid CNN-LSTM model")
    report.append("🌟 For real-time inference: Use CNN1D or Hybrid CNN-LSTM model")
    
    # Save report
    report_text = "\n".join(report)
    save_path = save_dir / 'evaluation_summary_report.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + "="*70)
    print(report_text)
    print("\n" + "="*70)
    print(f"Summary report saved to: {save_path}")

def main():
    """Main function to generate all visualizations"""
    print("ECG Anomaly Detection Model Visualization")
    print("=" * 50)
    
    # Load data
    df = load_model_results()
    if df is None:
        return
    
    # Create results directory
    save_dir = Path('results/visualization')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    print(f"Results will be saved to: {save_dir}")
    
    # Generate all visualizations
    create_performance_comparison(df, save_dir)
    create_radar_chart(df, save_dir)
    create_efficiency_analysis(df, save_dir)
    create_comprehensive_table(df, save_dir)
    create_inference_speed_analysis(df, save_dir)
    generate_summary_report(df, save_dir)
    
    print("\n" + "="*70)
    print("✅ All visualizations generated successfully!")
    print(f"📁 Check the '{save_dir}' directory for all generated files.")
    print("="*70)

if __name__ == '__main__':
    main()
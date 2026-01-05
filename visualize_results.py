#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize evaluation results summary
"""

import os
import sys
import pandas as pd
from pathlib import Path
import argparse

# Check dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required packages: {e}")
    print("Please run: pip install matplotlib seaborn")
    sys.exit(1)

# Set matplotlib style
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_summary_data(csv_file='summary_results.csv'):
    """Load summary data"""
    if not Path(csv_file).exists():
        raise FileNotFoundError(f"File {csv_file} does not exist. Please run summarize_results.py first.")
    
    df = pd.read_csv(csv_file)
    return df


def plot_metrics_comparison(df, output_dir='visualizations'):
    """
    Plot metrics comparison between models
    
    Parameters
    ----------
    df : DataFrame
        Summary data
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by model type
    metrics = ['MSE', 'MAE', 'RMSE', 'MAPE']
    
    # Filter out records with null MAPE
    df_plot = df.copy()
    df_plot = df_plot[df_plot['MAPE'].notna()] if 'MAPE' in df_plot.columns else df_plot
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        if metric not in df_plot.columns:
            continue
        
        ax = axes[idx]
        
        # Prepare data: group by dataset and model type
        pivot_data = df_plot.pivot_table(
            values=metric,
            index='dataset',
            columns='model_type',
            aggfunc='mean'
        )
        
        # Plot bar chart
        pivot_data.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.legend(title='Model Type', fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'metrics_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_metrics_by_dataset(df, output_dir='visualizations'):
    """
    Plot metrics comparison by dataset
    
    Parameters
    ----------
    df : DataFrame
        Summary data
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['MSE', 'MAE', 'RMSE']
    df_plot = df.copy()
    
    # Get all datasets
    datasets = df_plot['dataset'].unique()
    
    # Create chart for each dataset
    for dataset in datasets:
        dataset_df = df_plot[df_plot['dataset'] == dataset]
        
        if len(dataset_df) == 0:
            continue
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            if metric not in dataset_df.columns:
                continue
            
            ax = axes[idx]
            
            # Prepare data
            model_types = dataset_df['model_type'].values
            metric_values = dataset_df[metric].values
            model_names = dataset_df['model_name'].values
            
            # Create bar chart
            bars = ax.bar(range(len(model_types)), metric_values, 
                         color=['#3498db' if mt == 'teacher' else '#e74c3c' for mt in model_types])
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, metric_values)):
                if pd.notna(val):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=10)
            ax.set_xticks(range(len(model_types)))
            ax.set_xticklabels([f"{mt}\n{mn[:20]}" for mt, mn in zip(model_types, model_names)], 
                              rotation=45, ha='right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Dataset: {dataset}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Clean dataset name for filename
        safe_dataset = dataset.replace('/', '_').replace('\\', '_')[:50]
        output_file = Path(output_dir) / f'metrics_{safe_dataset}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def plot_heatmap(df, output_dir='visualizations'):
    """
    Plot heatmap: show metrics by dataset and model type
    
    Parameters
    ----------
    df : DataFrame
        Summary data
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['MSE', 'MAE', 'RMSE']
    df_plot = df.copy()
    
    for metric in metrics:
        if metric not in df_plot.columns:
            continue
        
        # Prepare data
        pivot_data = df_plot.pivot_table(
            values=metric,
            index='dataset',
            columns='model_type',
            aggfunc='mean'
        )
        
        if pivot_data.empty:
            continue
        
        # Create heatmap
        plt.figure(figsize=(10, max(8, len(pivot_data) * 0.5)))
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', 
                   cbar_kws={'label': metric}, linewidths=0.5)
        plt.title(f'{metric} Heatmap', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Model Type', fontsize=12)
        plt.ylabel('Dataset', fontsize=12)
        plt.tight_layout()
        
        output_file = Path(output_dir) / f'heatmap_{metric}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def plot_improvement_ratio(df, output_dir='visualizations'):
    """
    Plot improvement ratio of student model relative to teacher model
    
    Parameters
    ----------
    df : DataFrame
        Summary data
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['MSE', 'MAE', 'RMSE']
    df_plot = df.copy()
    
    # Group by dataset and calculate improvement ratio
    improvement_data = []
    
    for dataset in df_plot['dataset'].unique():
        dataset_df = df_plot[df_plot['dataset'] == dataset]
        
        teacher_df = dataset_df[dataset_df['model_type'] == 'teacher']
        student_df = dataset_df[dataset_df['model_type'] == 'student']
        
        if len(teacher_df) == 0 or len(student_df) == 0:
            continue
        
        teacher_metrics = teacher_df.iloc[0]
        student_metrics = student_df.iloc[0]
        
        for metric in metrics:
            if metric not in teacher_metrics or metric not in student_metrics:
                continue
            
            teacher_val = teacher_metrics[metric]
            student_val = student_metrics[metric]
            
            if pd.notna(teacher_val) and pd.notna(student_val) and teacher_val != 0:
                # Calculate improvement ratio (negative means worse, positive means better)
                improvement = ((teacher_val - student_val) / teacher_val) * 100
                improvement_data.append({
                    'dataset': dataset,
                    'metric': metric,
                    'improvement': improvement
                })
    
    if not improvement_data:
        print("Cannot calculate improvement ratio: missing teacher or student model data")
        return
    
    improvement_df = pd.DataFrame(improvement_data)
    
    # Plot improvement ratio
    fig, ax = plt.subplots(figsize=(14, 8))
    
    pivot_data = improvement_df.pivot_table(
        values='improvement',
        index='dataset',
        columns='metric',
        aggfunc='mean'
    )
    
    pivot_data.plot(kind='bar', ax=ax, width=0.8)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Student Model Improvement Ratio Relative to Teacher Model (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Improvement Ratio (%)', fontsize=12)
    ax.legend(title='Metric', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'improvement_ratio.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_summary_statistics(df, output_dir='visualizations'):
    """
    Plot summary statistics
    
    Parameters
    ----------
    df : DataFrame
        Summary data
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['MSE', 'MAE', 'RMSE']
    
    # Calculate average metrics by model type
    summary_stats = df.groupby('model_type')[metrics].mean()
    
    # Plot summary statistics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    summary_stats.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Average Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_ylabel('Average Metric Value', fontsize=12)
    ax.legend(title='Metric', fontsize=10)
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'summary_statistics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize evaluation results summary')
    parser.add_argument('--input', type=str, default='summary_results.csv', 
                       help='Input summary CSV file')
    parser.add_argument('--output_dir', type=str, default='visualizations', 
                       help='Output directory')
    parser.add_argument('--all', action='store_true', 
                       help='Generate all plots')
    parser.add_argument('--comparison', action='store_true', 
                       help='Generate metrics comparison plot')
    parser.add_argument('--by_dataset', action='store_true', 
                       help='Generate plots by dataset')
    parser.add_argument('--heatmap', action='store_true', 
                       help='Generate heatmap')
    parser.add_argument('--improvement', action='store_true', 
                       help='Generate improvement ratio plot')
    parser.add_argument('--summary', action='store_true', 
                       help='Generate summary statistics plot')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Starting visualization generation...")
    print("=" * 60)
    
    # Load data
    try:
        df = load_summary_data(args.input)
        print(f"\nSuccessfully loaded data: {len(df)} records")
        print(f"Number of datasets: {df['dataset'].nunique()}")
        print(f"Model types: {df['model_type'].unique()}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If no option specified or --all is specified, generate all plots
    if args.all or not any([args.comparison, args.by_dataset, args.heatmap, 
                            args.improvement, args.summary]):
        print("\nGenerating all plots...")
        plot_metrics_comparison(df, args.output_dir)
        plot_metrics_by_dataset(df, args.output_dir)
        plot_heatmap(df, args.output_dir)
        plot_improvement_ratio(df, args.output_dir)
        plot_summary_statistics(df, args.output_dir)
    else:
        if args.comparison:
            print("\nGenerating metrics comparison plot...")
            plot_metrics_comparison(df, args.output_dir)
        
        if args.by_dataset:
            print("\nGenerating plots by dataset...")
            plot_metrics_by_dataset(df, args.output_dir)
        
        if args.heatmap:
            print("\nGenerating heatmap...")
            plot_heatmap(df, args.output_dir)
        
        if args.improvement:
            print("\nGenerating improvement ratio plot...")
            plot_improvement_ratio(df, args.output_dir)
        
        if args.summary:
            print("\nGenerating summary statistics plot...")
            plot_summary_statistics(df, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Visualization completed!")
    print(f"All plots saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()


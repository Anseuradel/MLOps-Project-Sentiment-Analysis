"""
Simple EDA script without complex dependencies.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from src.model.data_extraction import load_data
import config

# Set basic style that works everywhere
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")


def load_real_data():
    """Load your actual dataset."""
    df = load_data(config.DATASET_PATH)
    return df


def plot_class_distribution(df, output_dir='outputs/eda'):
    """Plot beautiful class distribution charts."""
    os.makedirs(output_dir, exist_ok=True)
    
    class_counts = df['label_id'].value_counts().sort_index()
    sentiment_names = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    colors = ['#ff6b6b', '#ffa726', '#bdbdbd', '#66bb6a', '#2e7d32']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart
    bars = ax1.bar(sentiment_names, class_counts.values, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_title('Sentiment Class Distribution', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Sentiment Classes', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Pie chart
    wedges, texts, autotexts = ax2.pie(
        class_counts.values, 
        labels=sentiment_names,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11}
    )
    ax2.set_title('Class Distribution (%)', fontsize=16, fontweight='bold', pad=20)
    
    # Make percentage text white and bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Class distribution plot saved: {output_dir}/class_distribution.png")

def plot_text_analysis(df, output_dir='outputs/eda'):
    """Plot comprehensive text analysis."""
    # Calculate text statistics
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Text length distribution
    ax1.hist(df['text_length'], bins=8, alpha=0.7, color='skyblue', edgecolor='black', linewidth=1.2)
    ax1.set_title('Distribution of Text Length', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Text Length (characters)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(df['text_length'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {df["text_length"].mean():.1f} chars')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Word count distribution
    ax2.hist(df['word_count'], bins=8, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=1.2)
    ax2.set_title('Distribution of Word Count', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Word Count')
    ax2.set_ylabel('Frequency')
    ax2.axvline(df['word_count'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {df["word_count"].mean():.1f} words')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Text length by sentiment
    sentiment_data = []
    for label in sorted(df['label_id'].unique()):
        sentiment_name = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'][label]
        subset = df[df['label_id'] == label]
        for length in subset['text_length']:
            sentiment_data.append({'sentiment': sentiment_name, 'text_length': length})
    
    sentiment_df = pd.DataFrame(sentiment_data)
    sentiment_df.boxplot(column='text_length', by='sentiment', ax=ax3, 
                        patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
    ax3.set_title('Text Length by Sentiment', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Sentiment Class')
    ax3.set_ylabel('Text Length (characters)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(alpha=0.3)
    
    # 4. Word count by sentiment
    sentiment_wc_data = []
    for label in sorted(df['label_id'].unique()):
        sentiment_name = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'][label]
        subset = df[df['label_id'] == label]
        for count in subset['word_count']:
            sentiment_wc_data.append({'sentiment': sentiment_name, 'word_count': count})
    
    sentiment_wc_df = pd.DataFrame(sentiment_wc_data)
    sentiment_wc_df.boxplot(column='word_count', by='sentiment', ax=ax4,
                           patch_artist=True,
                           boxprops=dict(facecolor='lightcoral', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2))
    ax4.set_title('Word Count by Sentiment', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Sentiment Class')
    ax4.set_ylabel('Word Count')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(alpha=0.3)
    
    plt.suptitle('')  # Remove automatic title
    plt.tight_layout()
    plt.savefig(f'{output_dir}/text_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Text analysis plot saved: {output_dir}/text_analysis.png")

def generate_analysis_report(df, output_dir='outputs/eda'):
    """Generate comprehensive analysis report."""
    # Basic statistics
    total_samples = len(df)
    num_classes = df['label_id'].nunique()
    class_distribution = df['label_id'].value_counts().sort_index()
    
    # Text statistics
    text_lengths = df['text'].str.len()
    word_counts = df['text'].str.split().str.len()
    
    # Calculate imbalance ratio
    max_count = class_distribution.max()
    min_count = class_distribution.min()
    imbalance_ratio = max_count / min_count
    
    report = f"""
EXPLORATORY DATA ANALYSIS REPORT
{'=' * 50}

DATASET OVERVIEW:
- Total samples: {total_samples:,}
- Number of classes: {num_classes}
- Columns: {', '.join(df.columns)}
- Data types: {dict(df.dtypes)}

CLASS DISTRIBUTION ANALYSIS:
"""
    
    sentiment_names = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    for label, count in class_distribution.items():
        percentage = (count / total_samples) * 100
        report += f"- {sentiment_names[label]}: {count} samples ({percentage:.1f}%)\n"
    
    report += f"""
IMBALANCE ANALYSIS:
- Imbalance ratio (max/min): {imbalance_ratio:.2f}
- Majority class: {sentiment_names[class_distribution.idxmax()]}
- Minority class: {sentiment_names[class_distribution.idxmin()]}
- Dataset is {'highly imbalanced' if imbalance_ratio > 3 else 'moderately balanced' if imbalance_ratio > 1.5 else 'well balanced'}

TEXT STATISTICS:
- Average text length: {text_lengths.mean():.1f} characters
- Average word count: {word_counts.mean():.1f} words
- Shortest text: {text_lengths.min()} characters
- Longest text: {text_lengths.max()} characters
- Standard deviation: {text_lengths.std():.1f} characters

DATA QUALITY ASSESSMENT:
- Missing values: {df.isnull().sum().sum()}
- Duplicate texts: {df['text'].duplicated().sum()}
- Empty texts: {(df['text'].str.strip() == '').sum()}
- Unique texts: {df['text'].nunique()}

RECOMMENDATIONS:
"""
    
    if imbalance_ratio > 3:
        report += "- Consider using class weights in training\n- Try oversampling minority classes\n- Use stratified sampling for splits\n"
    else:
        report += "- Dataset is reasonably balanced for training\n- Standard training approach should work well\n"
    
    if text_lengths.max() > 500:
        report += "- Some texts are very long, consider truncation for BERT\n"
    
    report += "- Monitor model performance across all classes during training\n"
    
    # Save report
    with open(f'{output_dir}/eda_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Analysis report saved: {output_dir}/eda_report.txt")
    
    # Print summary to console
    print("\n" + "="*50)
    print("EDA SUMMARY")
    print("="*50)
    print(f"📊 Samples: {total_samples}")
    print(f"🎯 Classes: {num_classes}")
    print(f"⚖️  Imbalance ratio: {imbalance_ratio:.2f}")
    print(f"📝 Avg text length: {text_lengths.mean():.1f} chars")
    print(f"🔤 Avg word count: {word_counts.mean():.1f} words")
    print("="*50)

def main():
    """Run complete EDA analysis."""
    print("🚀 Starting Comprehensive EDA Analysis")
    print("=" * 50)
    
    # Create output directory
    output_dir = 'outputs/eda'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_real_data()
    print(f"📊 Loaded dataset with {len(df)} samples")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"🎯 Classes: {df['label_id'].nunique()}")
    
    # Generate all analyses
    print("\n📈 Generating visualizations...")
    plot_class_distribution(df, output_dir)
    plot_text_analysis(df, output_dir)
    generate_analysis_report(df, output_dir)
    
    print("\n✅ EDA Analysis Complete!")
    print(f"📁 All outputs saved to: {output_dir}/")
    print("\nGenerated files:")
    for file in os.listdir(output_dir):
        if file.endswith(('.png', '.txt')):
            print(f"  - {output_dir}/{file}")

if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.5)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

def plot_metrics(dataset_name, data_dict, filename):
    """
    Plots grouped bar charts for a specific dataset comparison.
    """
    metrics = ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'gleu', 'meteor']
    metric_labels = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'GLEU', 'METEOR']
    
    n_metrics = len(metrics)
    n_archs = len(data_dict)
    
    x = np.arange(n_metrics)
    # Width of individual bars
    width = 0.8 / n_archs
    
    colors = sns.color_palette("husl", n_archs)
    
    fig, ax = plt.subplots(figsize=(20, 12), dpi=150)
    ax.set_facecolor('#f8f9fa')
    
    for i, (arch_name, scores) in enumerate(data_dict.items()):
        # Calculate the offset for grouped bars
        pos = x + (i - (n_archs - 1) / 2) * width
        values = [scores[m] for m in metrics]
        bars = ax.bar(pos, values, width, label=arch_name, 
                     color=colors[i], edgecolor='white', linewidth=2.5,
                     alpha=0.85, zorder=3)
        
        # Add score labels on top of bars for precision
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14, rotation=45,
                        fontweight='bold', color='#2c3e50')

    ax.set_ylabel('Score Value', fontsize=20, fontweight='bold', color='#2c3e50')
    ax.set_xlabel('Evaluation Metrics', fontsize=20, fontweight='bold', color='#2c3e50')
    ax.set_title(f'Performance Metrics Comparison - {dataset_name} Dataset', 
                fontsize=24, fontweight='bold', pad=25, color='#1a252f')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=16, fontweight='bold')
    ax.tick_params(axis='y', labelsize=16)
    
    legend = ax.legend(title='Architecture', loc='upper right', 
                      frameon=True, shadow=True, fancybox=True,
                      fontsize=16, title_fontsize=18)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('#cccccc')
    legend.get_frame().set_linewidth(2)
    
    # Add extra headroom for labels
    max_val = max([max(d.values()) for d in data_dict.values()])
    ax.set_ylim(0, max_val * 1.2)
    
    # Refined grid
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#95a5a6', zorder=0, linewidth=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#34495e')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_color('#34495e')
    ax.spines['bottom'].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

flickr_data = {
    'CNN (ResNet-50)': {
        'bleu1': 0.2506, 'bleu2': 0.1339, 'bleu3': 0.0876, 'bleu4': 0.0613, 'gleu': 0.0958, 'meteor': 0.2042
    },
    'ViT (Base)': {
        'bleu1': 0.2320, 'bleu2': 0.1145, 'bleu3': 0.0735, 'bleu4': 0.0512, 'gleu': 0.0837, 'meteor': 0.1678
    },
    'CNN-CPTR (Hybrid)': {
        'bleu1': 0.2367, 'bleu2': 0.1217, 'bleu3': 0.0789, 'bleu4': 0.0554, 'gleu': 0.0882, 'meteor': 0.1770
    },
    'Random Baseline': {
        'bleu1': 0.0005, 'bleu2': 0.0004, 'bleu3': 0.0002, 'bleu4': 0.0002, 'gleu': 0.0002, 'meteor': 0.0086
    }
}

coco_data = {
    'CNN (ResNet-50)': {
        'bleu1': 0.3115, 'bleu2': 0.1946, 'bleu3': 0.1387, 'bleu4': 0.1045, 'gleu': 0.1362, 'meteor': 0.2544
    },
    'ViT (Base)': {
        'bleu1': 0.2353, 'bleu2': 0.1318, 'bleu3': 0.0900, 'bleu4': 0.0655, 'gleu': 0.0857, 'meteor': 0.1693
    },
    'Custom CPTR (Scratch)': {
        'bleu1': 0.2306, 'bleu2': 0.1254, 'bleu3': 0.0854, 'bleu4': 0.0617, 'gleu': 0.0810, 'meteor': 0.1558
    },
    'Random Baseline': {
        'bleu1': 0.0073, 'bleu2': 0.0053, 'bleu3': 0.0038, 'bleu4': 0.0027, 'gleu': 0.0019, 'meteor': 0.0103
    }
    
}

plot_metrics('FLICKR', flickr_data, 'results/plots/flickr_metrics_comparison.png')
plot_metrics('COCO', coco_data, 'results/plots/coco_metrics_comparison.png')
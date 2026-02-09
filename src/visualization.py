import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def plot_correlation_heatmap(matrix, categories, filename="figure2_heatmap.png"):
    """Plots and saves the correlation heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=categories, yticklabels=categories, vmin=-1, vmax=1)
    plt.title("Grand Average Correlation Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    logger.info(f"Saved heatmap to {filename}")
    plt.close()

def plot_exclusion_bars(drops_data, filename="figure4_exclusion.png"):
    """Plots and saves the exclusion analysis bar chart."""
    df = pd.DataFrame(drops_data)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Category", y="Drop", errorbar=('ci', 68), hue="Category", palette="muted", capsize=.1, legend=False)
    plt.title("Impact of Removing Preferred Voxels")
    plt.ylabel("Drop in Correlation (%)")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    plt.savefig(filename)
    logger.info(f"Saved bar plot to {filename}")
    plt.close()

def print_final_tables(accuracies, drops_data, subjects):
    """Logs formatted tables to the console."""
    df_acc = pd.DataFrame({'Subject': subjects, 'Accuracy (%)': accuracies})
    
    logger.info("\n" + "="*50)
    logger.info("TABLE 1: Classification Accuracy per Subject")
    logger.info("="*50)
    logger.info("\n" + df_acc.to_string(index=False, float_format="%.1f"))
    logger.info("-" * 50)
    logger.info(f"Group Mean Accuracy: {np.mean(accuracies):.1f}%")
    logger.info("="*50 + "\n")

    # Table 2: Exclusion Analysis
    df_drops = pd.DataFrame(drops_data)
    summary = df_drops.groupby("Category")[['Original Corr', 'Excluded Corr', 'Drop']].mean().reset_index()
    summary.columns = ['Category', 'Mean Orig Corr', 'Mean Excl Corr', 'Mean Drop (%)']
    
    logger.info("="*70)
    logger.info("TABLE 2: Detailed Correlation Drop Analysis (Mean across Subjects)")
    logger.info("="*70)
    logger.info("\n" + summary.to_string(index=False, float_format="%.3f"))
    logger.info("="*70 + "\n")
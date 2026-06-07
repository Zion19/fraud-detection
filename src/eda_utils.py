import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from IPython.display import display

def set_aesthetics(palette_style="muted", font_scale=1.1, figure_figsize=(12, 5)):
    """Sets the default plotting theme and options for EDA."""
    sns.set_theme(style="whitegrid", palette=palette_style, font_scale=font_scale)
    plt.rcParams.update({
        "figure.figsize": figure_figsize,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    })
    pd.set_option("display.float_format", "{:.4f}".format)
    print("Libraries loaded & global aesthetics set ✓")

def load_data(file_path):
    """Loads a dataset from CSV and prints basic shape and memory information."""
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory : {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
    return df

def assess_quality(df):
    """Reports missing values and duplicate rows in the DataFrame."""
    # Missing values
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(4)
    quality = pd.DataFrame({"missing_count": miss, "missing_%": miss_pct})
    quality = quality[quality.missing_count > 0]
    
    if quality.empty:
        print("No missing values detected.")
    else:
        print("Columns with missing values:")
        display(quality)
        
    # Duplicates
    n_dup = df.duplicated().sum()
    print(f"\nDuplicate rows : {n_dup:,}")
    return quality, n_dup

def plot_class_distribution(df, target_col, title, palette=None, save_path=None):
    """Plots class distribution as a bar chart and donut chart side-by-side."""
    if palette is None:
        palette = {0: "#4C72B0", 1: "#DD4444"}
        
    counts = df[target_col].value_counts().sort_index()
    pcts = (counts / counts.sum() * 100).round(4)
    
    # Text summary
    summary = pd.DataFrame({
        "count": counts.values,
        "percent_%": pcts.values
    }, index=[f"Normal (0)", f"Fraud (1)"] if len(counts) == 2 else [str(x) for x in counts.index])
    display(summary)
    
    if len(counts) == 2:
        ratio = counts.iloc[0] / counts.iloc[1]
        print(f"\nImbalance ratio (Normal : Fraud) = {ratio:.0f} : 1")
        
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Count bar
    ax = axes[0]
    labels = [f"Normal (0)", f"Fraud (1)"] if len(counts) == 2 else [str(x) for x in counts.index]
    bars = ax.bar(labels,
                  counts.values,
                  color=[palette.get(x, "#888888") for x in counts.index], 
                  edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=11)
    ax.set_title("Transaction Counts by Class", fontweight="bold")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    
    # Pie/Donut
    ax2 = axes[1]
    wedge_props = dict(width=0.55, edgecolor="white", linewidth=2)
    pie_labels = ["Normal", "Fraud"] if len(counts) == 2 else [str(x) for x in counts.index]
    ax2.pie(counts.values, labels=pie_labels,
            autopct="%1.3f%%", colors=[palette.get(x, "#888888") for x in counts.index],
            wedgeprops=wedge_props, startangle=90,
            textprops={"fontsize": 12})
    ax2.set_title("Class Proportion (Donut)", fontweight="bold")
    
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved class distribution plot to: {save_path}")
    plt.show()

def plot_correlation_heatmap(df, cols, title, cmap="coolwarm", save_path=None):
    """Plots a correlation matrix heatmap for numeric columns with lower-triangle mask."""
    corr_matrix = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap,
                vmin=-1, vmax=1, center=0,
                linewidths=0.3, linecolor="white",
                annot=False, ax=ax)
    ax.set_title(title, fontweight="bold", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved correlation heatmap to: {save_path}")
    plt.show()

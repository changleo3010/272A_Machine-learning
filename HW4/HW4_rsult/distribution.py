#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

"""
Distribution plotting utility for Task 2: Distribution Plots

- Loads diabetes.csv from the current working directory
- Automatically classifies features into Binary, Categorical, Numeric
- Generates and saves three composite plots:
  - binary.png (pie charts for binary features)
  - category.png (bar charts for categorical features)
  - numeric.png (histograms for numeric features)

Notes:
- The script does not modify diabetes.csv.
- Outputs are saved in the working directory (CWD).
"""

# Configurations
TOP_N_CATEGORIES = 15
HIST_BINS = 20

# Utility: load dataframe
def load_df(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}", flush=True)
        sys.exit(1)

# Classification rules (deterministic)
def classify_features(df, top_n=TOP_N_CATEGORIES):
    binary = []
    categorical = []
    numeric = []
    for col in df.columns:
        s = df[col]
        # number of non-null unique values
        n_unique = int(s.dropna().nunique())
        dtype = s.dtype
        is_numeric = pd.api.types.is_numeric_dtype(s)
        # Binary if exactly two unique non-null values
        if n_unique == 2:
            binary.append(col)
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            # Object or categorical: treat as categorical
            categorical.append(col)
        elif is_numeric:
            # Numeric with more than two unique values -> numeric, but keep high-cardinality numeric values as numeric
            if n_unique <= 20:
                # small cardinality numeric values can be treated as categorical for visualization purposes
                categorical.append(col)
            else:
                numeric.append(col)
        else:
            # Fallback to categorical for unknown types
            categorical.append(col)

    return {
        'binary': binary,
        'categorical': categorical,
        'numeric': numeric
    }

# Plot helpers

def plot_binary_pie_charts(df, binary_cols, outfile='binary.png'):
    if not binary_cols:
        print('No binary features detected; skipping binary plots.')
        return
    n = len(binary_cols)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    # Normalize axes to a flat list
    if rows == 1:
        axes = [axes]  # type: ignore
    axes = list(axes.flatten())

    for idx, col in enumerate(binary_cols):
        s = df[col]
        counts = s.dropna().value_counts()
        if counts.size < 2:
            ax = axes[idx]
            ax.text(0.5, 0.5, 'Not binary or insufficient data', ha='center', va='center')
            ax.set_title(col)
            ax.axis('off')
            continue
        labels = [str(v) for v in counts.index.tolist()]
        values = counts.values
        ax = axes[idx]
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title(col)
        ax.axis('equal')

    # Hide any unused axes
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)


def plot_categorical_bars(df, cat_cols, outfile='category.png', top_n=TOP_N_CATEGORIES):
    if not cat_cols:
        print('No categorical features detected; skipping category plots.')
        return
    n = len(cat_cols)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    if rows == 1:
        axes = [axes]  # type: ignore
    axes = list(axes.flatten())

    for idx, col in enumerate(cat_cols):
        s = df[col]
        counts = s.value_counts()
        if counts.empty:
            ax = axes[idx]
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(col)
            ax.axis('off')
            continue
        if len(counts) > top_n:
            top_counts = counts.iloc[:top_n].copy()
            others_sum = counts.iloc[top_n:].sum()
            top_counts.loc['Other'] = others_sum
            counts = top_counts
        labels = [str(v) for v in counts.index.tolist()]
        values = counts.values
        ax = axes[idx]
        color_map = plt.cm.tab20.colors
        bar_colors = [color_map[i % len(color_map)] for i in range(len(labels))]
        ax.bar(range(len(labels)), values, color=bar_colors)
        ax.set_title(col)
        ax.set_ylabel('Counts')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)


def plot_numeric_histograms(df, numeric_cols, outfile='numeric.png', bins=HIST_BINS):
    if not numeric_cols:
        print('No numeric features detected; skipping numeric plots.')
        return
    n = len(numeric_cols)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    if rows == 1:
        axes = [axes]  # type: ignore
    axes = list(axes.flatten())

    for idx, col in enumerate(numeric_cols):
        s = df[col].dropna()
        ax = axes[idx]
        if s.size == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(col)
            ax.axis('off')
            continue
        ax.hist(s, bins=bins, color=plt.cm.tab10(0))
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)

def main():
    csv_path = os.path.join(os.getcwd(), 'diabetes.csv')
    df = load_df(csv_path)
    if df is None or df.empty:
        print('No data to plot. Exiting.')
        return

    classification = classify_features(df)
    binary_cols = classification['binary']
    cat_cols = classification['categorical']
    num_cols = classification['numeric']

    print(f'Binary features ({len(binary_cols)}): {binary_cols}')
    print(f'Categorical features ({len(cat_cols)}): {cat_cols}')
    print(f'Numeric features ({len(num_cols)}): {num_cols}')

    plot_binary_pie_charts(df, binary_cols, outfile=os.path.join(os.getcwd(), 'binary.png'))
    plot_categorical_bars(df, cat_cols, outfile=os.path.join(os.getcwd(), 'category.png'))
    plot_numeric_histograms(df, num_cols, outfile=os.path.join(os.getcwd(), 'numeric.png'))

if __name__ == '__main__':
    main()

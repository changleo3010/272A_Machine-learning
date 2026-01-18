#!/usr/bin/env python3
"""corr.py
Load diabetes.csv, compute the full correlation matrix for all numeric features,
generate a heatmap, and save it as heatmap.png.
Additionally, compute correlations with Diabetes_binary, identify top 10 features,
and generate a bar chart saved as bar_chart.png.
"""

from __future__ import annotations

import sys
import os
from typing import Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_dataframe(csv_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def compute_full_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the full correlation matrix for numeric features only.

    Pandas DataFrame.corr() computes the correlation between numeric dtypes.
    Non-numeric columns are ignored in the calculation.
    """
    if df.empty:
        return df.copy()
    # Use Pearson correlation by default. You can switch to spearman if needed.
    return df.corr(method="pearson")


def generate_heatmap(corr_df: pd.DataFrame, heatmap_path: str, figsize: Optional[tuple] = (12, 10)) -> None:
    """Generate and save a heatmap from a correlation matrix."""
    if corr_df is None or corr_df.empty:
        raise ValueError("Correlation DataFrame is empty. Cannot generate heatmap.")

    plt.figure(figsize=figsize)
    # Heatmap with annotations for readability
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=.5)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()


def generate_bar_chart(full_corr: pd.DataFrame, target: str = "Diabetes_binary", bar_chart_path: str = "bar_chart.png", top_n: int = 10) -> None:
    """Generate a horizontal bar chart of the top_n features most correlated with the target."""
    if full_corr is None or full_corr.empty:
        raise ValueError("Correlation DataFrame is empty. Cannot generate bar chart.")
    if target not in full_corr.columns:
        raise ValueError(f"Target column '{target}' not found in correlation DataFrame.")

    abs_corr = full_corr[target].abs().drop(labels=[target], errors='ignore')
    if abs_corr.empty:
        raise ValueError("No features available for bar chart generation.")

    top_features = abs_corr.sort_values(ascending=False).head(top_n).index
    # Align values with the sign of the correlation for readability
    values = full_corr.loc[top_features, target]
    # Order by absolute correlation magnitude
    order = abs_corr.loc[top_features].sort_values(ascending=False).index
    values = values.loc[order]

    plt.figure(figsize=(8, max(4, len(order) * 0.4 + 1)))
    plt.barh(range(len(order)), values.values, color="tab:blue")
    plt.yticks(range(len(order)), order)
    plt.xlabel(f'Correlation with {target}')
    plt.title(f'Top {top_n} features correlated with {target}')
    plt.tight_layout()
    plt.savefig(bar_chart_path, dpi=300)
    plt.close()


def main(argv: Optional[list] = None) -> int:
    # Defaults
    csv_path = "diabetes.csv"
    heatmap_path = "heatmap.png"
    bar_chart_path = "bar_chart.png"

    if argv and len(argv) > 0:
        csv_path = argv[0]
    if argv and len(argv) > 1:
        heatmap_path = argv[1]
    if argv and len(argv) > 2:
        bar_chart_path = argv[2]

    try:
        df = load_dataframe(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return 1

    corr = compute_full_corr_matrix(df)

    # Generate heatmap for the full correlation matrix
    try:
        generate_heatmap(corr, heatmap_path)
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return 1

    # Generate bar chart for top 10 features correlated with Diabetes_binary
    try:
        generate_bar_chart(corr, target="Diabetes_binary", bar_chart_path=bar_chart_path, top_n=10)
    except Exception as e:
        print(f"Error generating bar chart: {e}")
        return 1

    print(f"Heatmap saved to {heatmap_path}. Correlation matrix shape: {corr.shape}")
    print(f"Bar chart saved to {bar_chart_path} (top 10 features correlated with Diabetes_binary).")
    return 0


if __name__ == "__main__":
    # Accept optional command line arguments: python corr.py [diabetes.csv] [heatmap.png] [bar_chart.png]
    sys.exit(main(sys.argv[1:]))

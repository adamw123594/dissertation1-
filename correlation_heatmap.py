import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# GitHub repo layout
DATA_PATH = Path('data/master_migration_economic_1991_2024.csv')
OUT_DIR = Path('figures'); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / 'correlation_heatmap.png'

def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if c != 'Year':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'Immigration' in df.columns and 'Emigration' in df.columns:
        df['Net_Migration'] = df['Immigration'] - df['Emigration']
    return df

def correlation_heatmap(df: pd.DataFrame, cols: list, save_path: Path):
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr(method='pearson')

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, aspect='equal')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson correlation', rotation=90)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.set_yticklabels(cols)

    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha='center', va='center')

    ax.set_title('Correlation Heatmap: Migration vs Economic Indicators')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved heatmap to {save_path}')
    return corr

if __name__ == '__main__':
    df = load_and_prepare(DATA_PATH)
    cols = ['Immigration','Emigration','Net_Migration','GDP','Unemployment','GBP_EUR','GBP_USD','Avg_House_Price','Population']
    corr = correlation_heatmap(df, cols, OUT_PATH)
    print(corr.round(3).to_string())

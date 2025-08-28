# knifecrime_vs_emigration.py
# --------------------------------
# Cleans knife-enabled crime data (fixes mislabelled 2014 as 2024),
# aggregates category totals, merges with emigration, and saves:
# - figures/emigration_vs_knifecrime_timeline.png
# - figures/emigration_vs_knifecrime_scatter.png
# - data/derived/knifecrime_emigration_2011_2024_clean.csv
# --------------------------------

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = Path('data')
FIG_DIR = Path('figures'); FIG_DIR.mkdir(parents=True, exist_ok=True)
DERIVED_DIR = Path('data/derived'); DERIVED_DIR.mkdir(parents=True, exist_ok=True)

KNIFE_CSV = DATA_DIR / 'knife_enabled_crime_uk.csv'
MIGRATION_CSV = DATA_DIR / 'master_migration_economic_1991_2024.csv'

CAT_COLS = ['Assault with injury and assault with intent to cause serious harm',
            'Robbery','Threats to kill','Other selected offences']

def load_and_clean_knife(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for c in CAT_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Knife_Crime_Total'] = df[CAT_COLS].sum(axis=1, skipna=True)
    df['Year'] = df['Year'].astype('Int64')
    # Fix duplicated 2024 standing in for 2014
    dup_2024_idx = df.index[(df['Year'] == 2024)].tolist()
    if 2014 not in df['Year'].dropna().astype(int).unique().tolist() and len(dup_2024_idx) >= 2:
        totals = df.loc[dup_2024_idx, 'Knife_Crime_Total']
        to_2014_idx = totals.idxmin()
        df.loc[to_2014_idx, 'Year'] = 2014
    out = df[['Year','Knife_Crime_Total']].dropna().astype({'Year':'int'}).sort_values('Year')
    return out

def load_emigration(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path)
    m.columns = m.columns.str.strip()
    m = m[['Year','Emigration']]
    return m

def make_plots(df: pd.DataFrame):
    # Timeline
    fig, ax1 = plt.subplots(figsize=(11,6))
    ax1.plot(df['Year'], df['Emigration'], color='tab:blue', marker='o', linewidth=2, label='Emigration (thousands)')
    ax1.set_xlabel('Year'); ax1.set_ylabel('Emigration (thousands)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue'); ax1.grid(True, linestyle='--', alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(df['Year'], df['Knife_Crime_Total'], color='tab:red', marker='s', linewidth=2, label='Knife Crime (offences)')
    ax2.set_ylabel('Knife-enabled crime (offences)', color='tab:red'); ax2.tick_params(axis='y', labelcolor='tab:red')
    plt.title('UK Emigration and Knife-enabled Crime Over Time (2011–2024)')
    plt.tight_layout(); plt.savefig(FIG_DIR / 'emigration_vs_knifecrime_timeline.png', dpi=300, bbox_inches='tight'); plt.close()

    # Scatter
    x = df['Knife_Crime_Total'].values.astype(float); y = df['Emigration'].values.astype(float)
    plt.figure(figsize=(9,6)); plt.scatter(x, y, alpha=0.8)
    if len(x) > 2:
        m, b = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 200)
        plt.plot(xs, m*xs + b, linewidth=2)
    plt.title('Emigration vs Knife-enabled Crime (2011–2024)')
    plt.xlabel('Knife-enabled crime (offences)'); plt.ylabel('Emigration (thousands)')
    plt.grid(True, linestyle='--', alpha=0.4); plt.tight_layout()
    plt.savefig(FIG_DIR / 'emigration_vs_knifecrime_scatter.png', dpi=300, bbox_inches='tight'); plt.close()

if __name__ == '__main__':
    knife = load_and_clean_knife(KNIFE_CSV)
    mig = load_emigration(MIGRATION_CSV)
    df = pd.merge(mig, knife, on='Year', how='inner')
    df.to_csv(DERIVED_DIR / 'knifecrime_emigration_2011_2024_clean.csv', index=False)
    make_plots(df)
    print('Saved figures and derived CSV.')
